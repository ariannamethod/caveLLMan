/*
 * cavellman.c — Self-evolving hieroglyphic language model
 *
 * Core: Transformer inference (88 glyph vocab)
 * + Hebbian plasticity (learns from every conversation, no backprop)
 * + Symbol emergence (proposes new hieroglyphs when patterns crystallize)
 * + Diffusion inference (MASK → iterative unmasking)
 *
 * The cave painter that teaches itself new signs.
 *
 * Build: make cavellman
 * Run:   ./cavellman --weights weights/cavellman_v3.bin --preset small
 *
 * Copyright (C) 2026 Arianna Method contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "notorch.h"
#include "semantic_tokenizer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/wait.h>

/* ── Limits ─────────────────────────────────────────────────────────────── */

#define MAX_VOCAB       256     /* 88 base + room for emerged symbols */
#define MAX_SEQ         128
#define MAX_EMERGED     128     /* max new symbols the model can create */
#define COOCCUR_SIZE    256     /* co-occurrence matrix dimension */
#define HEBBIAN_RANK    4       /* low-rank Hebbian LoRA (Sage 3: 8 too high for 96-dim) */
#define EMERGE_THRESHOLD 0.75f  /* co-occurrence threshold for emergence */
#define DISCIPLINE_WINDOW 50    /* interactions between emergence attempts */

/* ── Architecture (matches train_cavellman.c presets) ──────────────────── */

static int E     = 96;
static int H     = 8;
static int HD    = 12;
static int FFN_D = 384;
static int N_L   = 4;
static int CTX   = 128;

typedef struct { const char* name; int embd, heads, layers, ctx; } Preset;
static const Preset PRESETS[] = {
    {"tiny",     18, 3, 2, 32},
    {"micro",    48, 4, 3, 64},
    {"standard", 64, 8, 3, 64},
    {"small",    96, 8, 4, 128},
    {"medium",  128, 8, 4, 128},
};
#define N_PRESETS 5

/* Look up a preset by name; returns NULL if unknown. */
static const Preset* find_preset(const char* name) {
    if (!name) return NULL;
    for (int i = 0; i < N_PRESETS; i++)
        if (strcmp(PRESETS[i].name, name) == 0) return &PRESETS[i];
    return NULL;
}

/* Push preset dims into the globals (does not touch any model). */
static void set_globals_from_preset(const Preset* pr) {
    if (!pr) return;
    E = pr->embd; H = pr->heads; HD = E / H; FFN_D = 4 * E;
    N_L = pr->layers; CTX = pr->ctx;
}

/* Next-smaller preset by name. medium→small→standard→micro→tiny→tiny. */
static const char* smaller_preset_name(const char* pname) {
    for (int i = 1; i < N_PRESETS; i++)
        if (strcmp(PRESETS[i].name, pname) == 0) return PRESETS[i-1].name;
    return pname;   /* already at tiny, or unknown — no downsize */
}

/* ── Vocab (base 88 + emerged) ──────────────────────────────────────────── */

typedef struct {
    char tokens[MAX_VOCAB][32];
    int  vocab_size;      /* total (base + emerged) */
    int  base_size;       /* original 88 */
    int  bos_id;
    int  mask_id;
} CaveVocab;

/* ── Co-occurrence matrix (Hebbian surface layer) ───────────────────────── */

typedef struct {
    float matrix[COOCCUR_SIZE][COOCCUR_SIZE];   /* symmetric, decay 0.95/session */
    int   pair_count[COOCCUR_SIZE][COOCCUR_SIZE]; /* raw counts */
    int   total_interactions;
    int   last_emergence;   /* interaction count at last symbol creation */
} CoOccurrence;

/* ── Emerged symbol ─────────────────────────────────────────────────────── */

typedef struct {
    int  glyph_a;           /* first component glyph id */
    int  glyph_b;           /* second component glyph id */
    float strength;         /* co-occurrence strength at time of emergence */
    int  born_at;           /* interaction number */
    int  use_count;         /* times used since birth */
    int  alive;             /* 0 = dead (failed survival), 1 = alive, 2 = frozen (primitive) */
    int  depth;             /* 1 = base pair, 2+ = chain */
    char name[32];          /* auto-generated name: "glyph_a+glyph_b" */
} EmergedSymbol;

#define SURVIVAL_USES    5    /* must be used 5 times... */
#define SURVIVAL_WINDOW  500  /* ...within 500 interactions of birth, or die */
#define MAX_DEPTH        5    /* depth cap — beyond this, freeze as new primitive */

/* ── Model ──────────────────────────────────────────────────────────────── */

typedef struct {
    nt_tensor *rms1, *wq, *wk, *wv, *wo;
    nt_tensor *rms2, *w_fc1, *w_fc2;
    /* Hebbian LoRA adapters (per-layer) */
    float *heb_A_q, *heb_B_q;  /* [E×rank], [rank×E] */
    float *heb_A_v, *heb_B_v;
} Layer;

typedef struct {
    nt_tensor* wte;
    nt_tensor* wpe;
    Layer*     layers;
    nt_tensor* rms_f;
    nt_tensor* head;

    /* Per-cave architecture — caves in the colony can have different
     * sizes (smaller children grow out of same-size parents via mitosis
     * downsize). Every model_forward call swaps the global dim vars to
     * these values on entry so the existing helpers keep working. */
    int E, H, HD, FFN_D, N_L, CTX;

    /* Hebbian state */
    CoOccurrence cooccur;
    EmergedSymbol emerged[MAX_EMERGED];
    int n_emerged;
    float hebbian_lr;
    float hebbian_decay;
} CaveModel;

/* Helper: push/pop the global dim vars from a model. Every hot path
 * that reads E/N_L/... starts with DIMS_ENTER(m) and exits via
 * DIMS_LEAVE(). Safe because all model_forward / model_load calls
 * happen under g_learner.lock — single-threaded. */
#define DIMS_ENTER(m)   \
    int _saved_E = E, _saved_H = H, _saved_HD = HD, \
        _saved_FFN = FFN_D, _saved_NL = N_L, _saved_CTX = CTX; \
    E = (m)->E; H = (m)->H; HD = (m)->HD; \
    FFN_D = (m)->FFN_D; N_L = (m)->N_L; CTX = (m)->CTX

#define DIMS_LEAVE()    \
    E = _saved_E; H = _saved_H; HD = _saved_HD; \
    FFN_D = _saved_FFN; N_L = _saved_NL; CTX = _saved_CTX

/* ── Load vocab ─────────────────────────────────────────────────────────── */

static int load_vocab(const char* path, CaveVocab* v) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("Error: cannot open vocab %s\n", path); return -1; }
    char line[256];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    v->vocab_size = atoi(line);
    v->base_size = v->vocab_size;
    if (v->vocab_size <= 0 || v->vocab_size >= MAX_VOCAB - 2) {
        printf("Error: invalid vocab size %d\n", v->vocab_size);
        fclose(f); return -1;
    }
    for (int i = 0; i < v->vocab_size; i++) {
        if (!fgets(line, sizeof(line), f)) break;
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        strncpy(v->tokens[i], line, 31);
        v->tokens[i][31] = '\0';
    }
    v->bos_id = v->vocab_size;
    v->mask_id = v->vocab_size + 1;
    fclose(f);
    return 0;
}

static int vocab_find(CaveVocab* v, const char* tok) {
    for (int i = 0; i < v->vocab_size; i++)
        if (strcmp(v->tokens[i], tok) == 0) return i;
    return -1;
}

/* ── Co-occurrence tracking ─────────────────────────────────────────────── */

static void cooccur_init(CoOccurrence* co) {
    memset(co, 0, sizeof(CoOccurrence));
}

static void cooccur_update(CoOccurrence* co, int* tokens, int len) {
    /* Update co-occurrence for all pairs within window=3 */
    for (int i = 0; i < len; i++) {
        if (tokens[i] < 0 || tokens[i] >= COOCCUR_SIZE) continue;
        for (int j = i + 1; j < len && j <= i + 3; j++) {
            if (tokens[j] < 0 || tokens[j] >= COOCCUR_SIZE) continue;
            int a = tokens[i], b = tokens[j];
            co->pair_count[a][b]++;
            co->pair_count[b][a]++;
        }
    }
    co->total_interactions++;

    /* Recompute normalized matrix */
    int max_count = 1;
    for (int i = 0; i < COOCCUR_SIZE; i++)
        for (int j = 0; j < COOCCUR_SIZE; j++)
            if (co->pair_count[i][j] > max_count) max_count = co->pair_count[i][j];

    for (int i = 0; i < COOCCUR_SIZE; i++)
        for (int j = 0; j < COOCCUR_SIZE; j++)
            co->matrix[i][j] = (float)co->pair_count[i][j] / (float)max_count;
}

static void cooccur_decay(CoOccurrence* co, float factor) {
    for (int i = 0; i < COOCCUR_SIZE; i++)
        for (int j = 0; j < COOCCUR_SIZE; j++) {
            co->matrix[i][j] *= factor;
            /* Decay raw counts slowly */
            if (co->pair_count[i][j] > 0 && ((float)rand() / RAND_MAX) < (1.0f - factor))
                co->pair_count[i][j]--;
        }
}

/* ── Symbol emergence ───────────────────────────────────────────────────── */

/*
 * Check if a new symbol should emerge.
 * Rules:
 *   1. At least DISCIPLINE_WINDOW interactions since last emergence
 *   2. Co-occurrence between two base glyphs > EMERGE_THRESHOLD
 *   3. The pair hasn't already emerged
 *   4. Max MAX_EMERGED total emerged symbols
 */
static int try_emerge_symbol(CaveModel* model, CaveVocab* vocab) {
    CoOccurrence* co = &model->cooccur;

    if (model->n_emerged >= MAX_EMERGED) return -1;
    /* Birth is free — survival is not (checked in check_symbol_survival) */

    /* Find strongest non-emerged pair */
    float best_score = 0;
    int best_a = -1, best_b = -1;

    for (int i = 0; i < vocab->base_size; i++) {
        for (int j = i + 1; j < vocab->base_size; j++) {
            if (co->matrix[i][j] < EMERGE_THRESHOLD) continue;
            if (co->matrix[i][j] <= best_score) continue;

            /* Check not already emerged */
            int already = 0;
            for (int k = 0; k < model->n_emerged; k++) {
                if ((model->emerged[k].glyph_a == i && model->emerged[k].glyph_b == j) ||
                    (model->emerged[k].glyph_a == j && model->emerged[k].glyph_b == i)) {
                    already = 1; break;
                }
            }
            if (already) continue;

            best_score = co->matrix[i][j];
            best_a = i; best_b = j;
        }
    }

    if (best_a < 0) return -1;

    /* Compute depth: if either parent is emerged, depth = max(parent depths) + 1 */
    int depth = 1;
    for (int k = 0; k < model->n_emerged; k++) {
        int eid = vocab->base_size + k;
        if ((eid == best_a || eid == best_b) && model->emerged[k].alive) {
            if (model->emerged[k].depth + 1 > depth)
                depth = model->emerged[k].depth + 1;
        }
    }
    if (depth > MAX_DEPTH) return -1; /* too deep — don't create */

    /* Emerge! */
    EmergedSymbol* sym = &model->emerged[model->n_emerged];
    sym->glyph_a = best_a;
    sym->glyph_b = best_b;
    sym->strength = best_score;
    sym->born_at = co->total_interactions;
    sym->use_count = 0;
    sym->alive = 1;
    sym->depth = depth;
    snprintf(sym->name, 32, "%s+%s", vocab->tokens[best_a], vocab->tokens[best_b]);

    int new_id = vocab->vocab_size;
    strncpy(vocab->tokens[new_id], sym->name, 31);
    vocab->tokens[new_id][31] = '\0';
    vocab->vocab_size++;

    model->n_emerged++;
    co->last_emergence = co->total_interactions;

    printf("\n  *** SYMBOL EMERGED: %s (id=%d, depth=%d, strength=%.3f) ***\n\n",
           sym->name, new_id, depth, best_score);

    return new_id;
}

/*
 * Symbol survival check — evolution needs death.
 *
 * Emerged symbols live in the vocab as metadata, not as trained embeddings —
 * the transformer head still outputs only the 88 base glyphs + BOS/MASK.
 * That means `use_count` is rarely incremented (only when the model happens
 * to emit an emerged id via generate, which is extremely unlikely), so the
 * old "die if use_count < SURVIVAL_USES" rule killed 100% of symbols on
 * passive reading (confirmed: Dracula → 9 born, 9 died with 0 uses).
 *
 * Biological fix: a symbol lives while its parent pair is still co-occurring
 * strongly in the surface layer. When the pattern that produced it fades
 * (co-occ < EMERGE_THRESHOLD * 0.7), the symbol dies. If the pattern stays
 * strong and the symbol reached MAX_DEPTH, it freezes into a new primitive.
 * Use_count is kept as a bonus signal — any actual usage (via generate)
 * extends life.
 */
static void check_symbol_survival(CaveModel* model, CaveVocab* vocab) {
    (void)vocab;
    int now = model->cooccur.total_interactions;
    CoOccurrence* co = &model->cooccur;
    float revival_floor = EMERGE_THRESHOLD * 0.7f;

    for (int i = 0; i < model->n_emerged; i++) {
        EmergedSymbol* sym = &model->emerged[i];
        if (sym->alive != 1) continue; /* skip dead or frozen */

        int age = now - sym->born_at;
        if (age < SURVIVAL_WINDOW) continue;

        float current_cooccur = (sym->glyph_a < COOCCUR_SIZE && sym->glyph_b < COOCCUR_SIZE)
            ? co->matrix[sym->glyph_a][sym->glyph_b] : 0.0f;
        int pattern_alive = (current_cooccur >= revival_floor);
        int got_used     = (sym->use_count >= SURVIVAL_USES);

        if (!pattern_alive && !got_used) {
            sym->alive = 0;
            printf("  *** SYMBOL DIED: %s (co-occ %.2f < %.2f, uses %d/%d, age %d) ***\n",
                   sym->name, current_cooccur, revival_floor,
                   sym->use_count, SURVIVAL_USES, age);
        } else if (sym->depth >= MAX_DEPTH) {
            sym->alive = 2;
            printf("  *** SYMBOL FROZEN: %s → new primitive (depth %d, co-occ %.2f, uses %d) ***\n",
                   sym->name, sym->depth, current_cooccur, sym->use_count);
        }
    }
}

/* Count emerged symbol usage in a token sequence */
static void count_emerged_usage(CaveModel* model, CaveVocab* vocab, int* tokens, int len) {
    for (int t = 0; t < len; t++) {
        int id = tokens[t];
        if (id >= vocab->base_size && id < vocab->base_size + model->n_emerged) {
            model->emerged[id - vocab->base_size].use_count++;
        }
    }
}

/* ── Hebbian LoRA allocation ────────────────────────────────────────────── */

static void alloc_hebbian(Layer* layer) {
    layer->heb_A_q = (float*)calloc(E * HEBBIAN_RANK, sizeof(float));
    layer->heb_B_q = (float*)calloc(HEBBIAN_RANK * E, sizeof(float));
    layer->heb_A_v = (float*)calloc(E * HEBBIAN_RANK, sizeof(float));
    layer->heb_B_v = (float*)calloc(HEBBIAN_RANK * E, sizeof(float));
}

/* ── Model loading ──────────────────────────────────────────────────────── */

static CaveModel* model_load(const char* weights_path, const Preset* pr) {
    if (!pr) {
        printf("Error: model_load called with NULL preset\n");
        return NULL;
    }
    /* Push this preset's dims into the globals before anything allocates
     * against them (alloc_hebbian uses E, the expected-tensor count uses
     * N_L, etc.). Caller does NOT need to swap first. */
    set_globals_from_preset(pr);

    int n_loaded = 0;
    nt_tensor** loaded = nt_load(weights_path, &n_loaded);
    if (!loaded || n_loaded == 0) {
        printf("Error: failed to load weights from %s\n", weights_path);
        return NULL;
    }

    int expected = 2 + N_L * 8 + 2;
    if (n_loaded < expected) {
        printf("Error: weight file has %d tensors, need %d\n", n_loaded, expected);
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded); return NULL;
    }

    CaveModel* m = (CaveModel*)calloc(1, sizeof(CaveModel));
    m->layers = (Layer*)calloc(N_L, sizeof(Layer));
    m->hebbian_lr = 0.001f;
    m->hebbian_decay = 0.9999f;

    /* Freeze this cave's dims. Future model_forward / hebbian_update swap
     * the globals back to these on entry — so several caves with different
     * presets can coexist in the ring. */
    m->E = E; m->H = H; m->HD = HD;
    m->FFN_D = FFN_D; m->N_L = N_L; m->CTX = CTX;

    int pi = 0;
    m->wte = loaded[pi++];
    m->wpe = loaded[pi++];
    for (int l = 0; l < N_L; l++) {
        m->layers[l].rms1   = loaded[pi++];
        m->layers[l].wq     = loaded[pi++];
        m->layers[l].wk     = loaded[pi++];
        m->layers[l].wv     = loaded[pi++];
        m->layers[l].wo     = loaded[pi++];
        m->layers[l].rms2   = loaded[pi++];
        m->layers[l].w_fc1  = loaded[pi++];
        m->layers[l].w_fc2  = loaded[pi++];
        alloc_hebbian(&m->layers[l]);
    }
    m->rms_f = loaded[pi++];
    m->head  = loaded[pi++];

    cooccur_init(&m->cooccur);

    printf("  loaded %d/%d tensors, Hebbian rank=%d, preset=%s (E=%d L=%d)\n",
           pi, n_loaded, HEBBIAN_RANK, pr->name, E, N_L);
    free(loaded);
    return m;
}

/* ── Forward pass (single token, with KV cache + Hebbian) ──────────────── */

/* KV cache — global static. With g_learner.lock held through dual_generate
 * in async mode, only one cave thread is in forward at a time, so a shared
 * KV cache is safe. We tried __thread (TLS) for parallel matvecs across
 * cores, but that increased per-thread memory by ~1 MB; on the Railway
 * trinity deploy 3 cave threads × 1 MB TLS pushed something over a limit
 * and the container died silently before printing the post-generate
 * debug line. Reverting to globals — correctness over micro-optimization. */
static float kv_keys[8][128][128];   /* [layer][pos][E] */
static float kv_vals[8][128][128];
static int   kv_len = 0;

static void kv_reset(void) { kv_len = 0; }

static void rmsnorm(float* out, const float* x, const float* gamma, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * gamma[i];
}

static void matvec(float* out, const float* W, const float* x, int nout, int nin) {
    /* BLAS fast path: on Accelerate/OpenBLAS this is cblas_sgemv; without
     * USE_BLAS it falls back to the same naive nested loop we used to have.
     * Medium preset (E=128, L=4, H=8): 4 QKVO × 128² + 2 FFN × 128×512 +
     * head × V×128 per token — on pure C that was several ms/token; sgemv
     * drops it to tens of μs. At 0.1s tick rate that's the difference
     * between idle cave and talking cave. */
    nt_blas_matvec(out, W, x, nout, nin);
}

/* Apply Hebbian LoRA: out += B @ (A @ x) */
static void apply_hebbian_lora(float* out, const float* A, const float* B,
                                const float* x, int dim, int rank) {
    float proj[64]; /* rank <= 64 */
    for (int r = 0; r < rank; r++) {
        float s = 0;
        for (int j = 0; j < dim; j++) s += A[j * rank + r] * x[j];
        proj[r] = s;
    }
    for (int i = 0; i < dim; i++) {
        float s = 0;
        for (int r = 0; r < rank; r++) s += B[r * dim + i] * proj[r];
        out[i] += s;
    }
}

static float* model_forward(CaveModel* m, int token_id, int pos) {
    /* Swap the dim globals to this cave's architecture so helpers (matvec,
     * rmsnorm, KV cache indexing) operate on the right shapes. Restored
     * at the single exit point below. */
    DIMS_ENTER(m);
    float x[256], xn[256], q[256], k[256], v[256];
    float attn_out[256], fc1[1024], fc2[256], proj[256];

    /* Clamp token_id to wte's actual row count. vocab_size can exceed the
     * file's saved V (emerged composites live in vocab but not in wte),
     * so sampling may hand us an id past the tensor — bind it to the last
     * representable row rather than reading heap garbage. Same for pos
     * vs. wpe — downsized children have a smaller CTX than their parents'
     * last spoken prompt, and pos can outrun wpe's saved row count. */
    int wte_rows = m->wte->len / E;
    int wpe_rows = m->wpe->len / E;
    int safe_id  = (token_id >= 0 && token_id < wte_rows) ? token_id : 0;
    int safe_pos = (pos >= 0 && pos < wpe_rows) ? pos : wpe_rows - 1;

    /* Embed */
    for (int j = 0; j < E; j++)
        x[j] = m->wte->data[safe_id * E + j] + m->wpe->data[safe_pos * E + j];

    for (int l = 0; l < N_L; l++) {
        Layer* ly = &m->layers[l];

        /* RMSNorm + QKV */
        rmsnorm(xn, x, ly->rms1->data, E);
        matvec(q, ly->wq->data, xn, E, E);
        matvec(k, ly->wk->data, xn, E, E);
        matvec(v, ly->wv->data, xn, E, E);

        /* Hebbian LoRA on Q and V */
        apply_hebbian_lora(q, ly->heb_A_q, ly->heb_B_q, xn, E, HEBBIAN_RANK);
        apply_hebbian_lora(v, ly->heb_A_v, ly->heb_B_v, xn, E, HEBBIAN_RANK);

        /* Store KV — use safe_pos so children with smaller CTX don't write
         * past the kv cache or their own usable window. */
        memcpy(kv_keys[l][safe_pos], k, E * sizeof(float));
        memcpy(kv_vals[l][safe_pos], v, E * sizeof(float));

        /* Multi-head attention */
        memset(attn_out, 0, E * sizeof(float));
        float scale = 1.0f / sqrtf((float)HD);
        for (int h = 0; h < H; h++) {
            int hs = h * HD;
            float scores[128];
            for (int t = 0; t <= safe_pos; t++) {
                float dot = 0;
                for (int j = 0; j < HD; j++) dot += q[hs+j] * kv_keys[l][t][hs+j];
                scores[t] = dot * scale;
            }
            /* Softmax */
            float mx = scores[0];
            for (int t = 1; t <= safe_pos; t++) if (scores[t] > mx) mx = scores[t];
            float sm = 0;
            for (int t = 0; t <= safe_pos; t++) { scores[t] = expf(scores[t] - mx); sm += scores[t]; }
            for (int t = 0; t <= safe_pos; t++) scores[t] /= sm;
            /* Weighted sum */
            for (int j = 0; j < HD; j++) {
                float val = 0;
                for (int t = 0; t <= safe_pos; t++) val += scores[t] * kv_vals[l][t][hs+j];
                attn_out[hs+j] = val;
            }
        }

        /* Project + residual */
        matvec(proj, ly->wo->data, attn_out, E, E);
        for (int j = 0; j < E; j++) x[j] += proj[j];

        /* FFN */
        rmsnorm(xn, x, ly->rms2->data, E);
        matvec(fc1, ly->w_fc1->data, xn, FFN_D, E);
        for (int j = 0; j < FFN_D; j++) fc1[j] = fc1[j] / (1.0f + expf(-fc1[j])); /* SiLU */
        matvec(fc2, ly->w_fc2->data, fc1, E, FFN_D);
        for (int j = 0; j < E; j++) x[j] += fc2[j];
    }

    /* Final norm + head */
    rmsnorm(xn, x, m->rms_f->data, E);
    /* Output vocab dimension comes from the head tensor on disk — not MAX_VOCAB.
     * Reading MAX_VOCAB rows out of a V-sized head walks past the tensor,
     * which is benign on some release-build heap layouts and a segfault on
     * others. Always read exactly what the tensor contains. */
    int Vp_actual = m->head->len / E;
    if (Vp_actual > MAX_VOCAB) Vp_actual = MAX_VOCAB;
    float* logits = (float*)calloc(MAX_VOCAB, sizeof(float));
    matvec(logits, m->head->data, xn, Vp_actual, E);
    DIMS_LEAVE();
    return logits;
}

/* ── Hebbian update after generation ────────────────────────────────────── */

/*
 * Compute prediction error signal for a token:
 * how surprised was the model? ||embedding_predicted - embedding_actual|| / sqrt(E)
 * Normalized to [0.1, 2.0]. Floor at 0.1 — even boring tokens leave a trace.
 */
static float prediction_error_signal(CaveModel* m, float* logits, int actual_id, int vocab_size) {
    /* Softmax to get predicted distribution */
    float mx = logits[0];
    for (int i = 1; i < vocab_size; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0;
    for (int i = 0; i < vocab_size; i++) sum += expf(logits[i] - mx);
    float predicted_prob = expf(logits[actual_id] - mx) / sum;

    /* Signal = -log(p) normalized. High surprise = high signal */
    float surprise = -logf(predicted_prob + 1e-8f);
    float signal = surprise / logf((float)vocab_size); /* normalize by max possible surprise */
    if (signal < 0.1f) signal = 0.1f;
    if (signal > 2.0f) signal = 2.0f;
    return signal;
}

static void hebbian_update(CaveModel* m, float* last_logits, int vocab_size,
                           int* generated, int gen_len, int is_passive) {
    DIMS_ENTER(m);
    for (int l = 0; l < N_L; l++) {
        Layer* ly = &m->layers[l];
        for (int g = 0; g < gen_len; g++) {
            /* Prediction error signal — learn more from surprise */
            float signal = (last_logits && g == gen_len - 1)
                ? prediction_error_signal(m, last_logits, generated[g], vocab_size)
                : 1.0f;

            /* Passive reading (ingested text) = 0.3x, active conversation = 1.0x */
            if (is_passive) signal *= 0.3f;

            /* Boost for emerged symbols — reinforce what the system discovered */
            if (generated[g] >= m->cooccur.total_interactions) signal *= 1.5f;

            /* Clamp the token index to the loaded wte row count, same as
             * model_forward does. Emerged symbols can grow vocab past the
             * .bin's V (86 + emerged), and sampling sometimes returns those
             * emerged IDs — without the clamp hebbian_update reads past the
             * end of wte->data and SIGSEGVs. Trinity's Molly thread emerged
             * 6 symbols in 3s on Linux Railway and crashed exactly here. */
            long wte_rows = m->wte->len / E;
            int safe_id = (generated[g] >= 0 && (long)generated[g] < wte_rows)
                ? generated[g] : 0;
            float* x_emb = m->wte->data + (long)safe_id * E;
            float* dy = x_emb;

            /* Active conversation: update Q + V. Passive: V only */
            if (!is_passive) {
                nt_hebbian_step(ly->heb_A_q, ly->heb_B_q, E, E, HEBBIAN_RANK,
                               x_emb, dy, signal, m->hebbian_lr, m->hebbian_decay);
            }
            nt_hebbian_step(ly->heb_A_v, ly->heb_B_v, E, E, HEBBIAN_RANK,
                           x_emb, dy, signal, m->hebbian_lr, m->hebbian_decay);
        }
    }
    DIMS_LEAVE();
}

/* ── Top-p sampling ─────────────────────────────────────────────────────── */

static int sample_top_p(float* logits, int n, float temp, float top_p) {
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float probs[MAX_VOCAB];
    float sum = 0;
    for (int i = 0; i < n; i++) { probs[i] = expf((logits[i] - mx) / temp); sum += probs[i]; }
    for (int i = 0; i < n; i++) probs[i] /= sum;

    int indices[MAX_VOCAB];
    for (int i = 0; i < n; i++) indices[i] = i;
    for (int i = 1; i < n; i++) {
        int key = indices[i]; float kp = probs[key]; int j = i - 1;
        while (j >= 0 && probs[indices[j]] < kp) { indices[j+1] = indices[j]; j--; }
        indices[j+1] = key;
    }

    float cum = 0; int nuc = 0;
    for (int i = 0; i < n; i++) { cum += probs[indices[i]]; nuc++; if (cum >= top_p) break; }
    float nuc_sum = 0;
    for (int i = 0; i < nuc; i++) nuc_sum += probs[indices[i]];
    float r = (float)rand() / (float)RAND_MAX;
    float c = 0;
    for (int i = 0; i < nuc; i++) { c += probs[indices[i]] / nuc_sum; if (c >= r) return indices[i]; }
    return indices[0];
}

/* ── Tokenize prompt ────────────────────────────────────────────────────── */

static int tokenize_prompt(const char* input, CaveVocab* vocab, int* tokens, int max) {
    int n = 0;
    char buf[256];
    strncpy(buf, input, 255); buf[255] = '\0';
    char* tok = strtok(buf, " \t\n");
    while (tok && n < max) {
        int id = vocab_find(vocab, tok);
        if (id >= 0) tokens[n++] = id;
        tok = strtok(NULL, " \t\n");
    }
    return n;
}

/* Persistence layer — binary per-cave spore with fingerprint + CRC —
 * lives at end of file, just before main(). See the persistence section
 * below and memory/project_cavellman_persistence.md for the design note. */

/* ── Ring physics types ─────────────────────────────────────────────────
 *
 * A ring of N caves (COLONY_MAX = 8 hard cap on Mac-scale). Each cave
 * carries a CaveField — somatic accumulators (excitement, dissonance),
 * a drifting coherence_floor (Stanley's silence-gate), and mass counters
 * for the Arianna-style notorch microtrain CPT. The founders A and B
 * just happen to be the first two; everything else (mitosis, death, DNA)
 * treats them symmetrically.
 */

#define TUNNEL_THRESHOLD    0.40f  /* dissonance → forced speech */
#define EXCITEMENT_DECAY    0.94f
#define DISSONANCE_DECAY    0.90f
#define EXCITEMENT_CAP      2.5f
#define MATURITY_WINDOW     40
#define MATURITY_STEP       0.005f
#define MATURITY_CAP        0.30f
#define SPEAK_RATIO_HIGH    0.70f
#define SPEAK_RATIO_LOW     0.20f
#define DUAL_TICK_US        100000 /* 0.1s per tick — tightened 2026-04-22 from 0.4s for livelier ring */
#define DUAL_MAX_GEN        24

#define MICRO_MIN_BYTES     2500
#define MICRO_MIN_NOVELTY   8.0f
#define MICRO_MIN_RESONANCE 15.0f
#define MICRO_TRAIN_STEPS   "300"

#define COLONY_MAX   8
#define DNA_DIR      "dna"
#define DNA_MAX_AGE  3600     /* seconds — older .txt files in dna/ expire */

typedef struct {
    float excitement;
    float coherence_floor;
    float baseline_floor;
    float dissonance;
    int   spoke_count;
    int   total_count;
    const char* name;

    /* Mass threshold CPT (Arianna-style) */
    long  mass_bytes;
    float mass_novelty;
    float mass_resonance;
    char  holding_path[512];    /* feed/<name>_holding.txt */
    char  weights_path[512];    /* current on-disk .bin for --start-from */
    char  next_weights_path[512]; /* where child will save */
    const char* preset_name;    /* pass to child */
    int   microtrain_active;
    pid_t microtrain_pid;
    int   microtrain_done_count;

    /* Pressure death: newborns get N ticks of immunity so they're not
     * culled before having a chance to prove themselves in the ring. */
    int   immunity_ticks;
} CaveField;

typedef struct {
    CaveModel* model;
    CaveVocab* vocab;
    CaveField  field;
    int        owns_vocab;
    int        is_founder;   /* A and B (and M in trinity mode) — shielded from pressure death forever */
    int        is_lover;     /* Molly — physically a founder in ring, semantically "external desire". */
    int        is_bastard;   /* Born of cross-mitosis lover × non-lover. Triggers jealousy field event. */
    /* Persistence — "yesterday" on this cave's mind */
    int     last_tokens[32]; /* what was on mind when cave slept; 32 = CAVE_LAST_TOKENS */
    int     last_len;        /* 0..32 */
    int64_t spore_saved_at;  /* 0 for cold-start cave, else header timestamp */
} Cave;

static Cave* g_colony[COLONY_MAX];
static int   g_colony_n = 0;

/* Forward declarations for the persistence layer (defined just above main).
 * cave_new / colony_main / colony_mitosis call these to hydrate / save /
 * blend spore state. */
static uint64_t cave_weights_fingerprint(nt_tensor** tensors, int n);
static int      load_cave_spore(Cave* c, uint64_t expected_fingerprint,
                                int* last_tokens_out, int* last_len_out);
static int      save_cave_spore(const Cave* c, const int* last_tokens,
                                int last_len, uint64_t fingerprint);
static float    cave_wake_impulse(Cave* c, int64_t saved_at);
static int      blend_spores(const Cave* pa, const Cave* pb, Cave* child);

/* Helper: compute this cave's weights fingerprint from the already-loaded
 * CaveModel tensors. */
static uint64_t cave_fingerprint_of(const Cave* c) {
    if (!c || !c->model) return 0;
    nt_tensor* fp[4] = {
        c->model->wte, c->model->wpe,
        c->model->layers[0].rms1, c->model->layers[0].wq
    };
    return cave_weights_fingerprint(fp, 4);
}

/* ── Async self-learning thread ─────────────────────────────────────────── */

/*
 * The colony shares two text streams:
 *   feed/  — external nudge (human drops a .txt here; all caves hear)
 *   dna/   — the colony's own generated text; caves write their output
 *             here and each other consume it on the next sweep.
 * A single learner thread watches both. Every active cave absorbs
 * whatever is dropped or produced — passive reading, 0.3× signal, V-only.
 * Old .txt in dna/ expire after DNA_MAX_AGE so the pool forgets as well
 * as remembers.
 */
typedef struct {
    const char* feed_dir;
    const char* dna_dir;
    int         running;
    int         files_consumed;
    int         lines_learned;
    pthread_mutex_t lock;
} AsyncLearner;

/*
 * Split text into sentences (phonons, per SPA from Q).
 * Tokens are atoms. Sentences are phonons. — Ландау's invention.
 * Splits on .!? followed by space/newline/EOF.
 */
static void learn_from_text(AsyncLearner* al, const char* text, int text_len) {
    char* buf = (char*)malloc(text_len + 1);
    memcpy(buf, text, text_len); buf[text_len] = '\0';

    /* Sentence splitting: .!? followed by space/newline/end */
    int sent_start = 0;
    for (int i = 0; i <= text_len; i++) {
        int is_boundary = (i == text_len);
        if (!is_boundary && (buf[i] == '.' || buf[i] == '!' || buf[i] == '?')) {
            /* Check next char is space, newline, quote, or end */
            int next = (i + 1 < text_len) ? buf[i + 1] : ' ';
            if (next == ' ' || next == '\n' || next == '\r' || next == '"' ||
                next == '\'' || next == ')' || i + 1 >= text_len)
                is_boundary = 1;
        }
        if (!is_boundary) continue;

        /* Extract sentence */
        int end = (i < text_len) ? i + 1 : i;
        int len = end - sent_start;
        if (len < 3) { sent_start = end; continue; }

        char sentence[4096];
        if (len >= 4096) len = 4095;
        memcpy(sentence, buf + sent_start, len);
        sentence[len] = '\0';
        sent_start = end;

        /* Skip whitespace-only sentences */
        int has_alpha = 0;
        for (int j = 0; j < len; j++)
            if ((sentence[j] >= 'a' && sentence[j] <= 'z') ||
                (sentence[j] >= 'A' && sentence[j] <= 'Z')) { has_alpha = 1; break; }
        if (!has_alpha) continue;

        int tokens[MAX_SEQ];
        int n = semtok_line(sentence, tokens, MAX_SEQ - 1);
        if (n >= 2) {
            pthread_mutex_lock(&al->lock);

            /* Every cave in the colony hears the sentence — passive, 0.3×, V-only. */
            for (int ci = 0; ci < g_colony_n; ci++) {
                Cave* cv = g_colony[ci];
                if (!cv) continue;
                hebbian_update(cv->model, NULL, cv->vocab->vocab_size + 2, tokens, n, 1);
                count_emerged_usage(cv->model, cv->vocab, tokens, n);
                cooccur_update(&cv->model->cooccur, tokens, n);
                check_symbol_survival(cv->model, cv->vocab);
                try_emerge_symbol(cv->model, cv->vocab);
            }

            al->lines_learned++;
            pthread_mutex_unlock(&al->lock);
        }
    }
    free(buf);
}

/*
 * One sweep through a directory — learn from any fresh .txt (except engines'
 * _holding.txt buffers), rename to .learned. If `expire` is true, delete any
 * .txt (or .learned) older than DNA_MAX_AGE so the DNA pool forgets.
 */
static void learner_scan_dir(AsyncLearner* al, const char* dir_path,
                             const char* tag, int expire) {
    DIR* dir = opendir(dir_path);
    if (!dir) return;
    time_t now = time(NULL);

    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (!al->running) break;
        int nlen = (int)strlen(ent->d_name);
        if (ent->d_name[0] == '.') continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir_path, ent->d_name);

        /* TTL expiration: drop anything older than DNA_MAX_AGE in this dir */
        if (expire) {
            struct stat st;
            if (stat(path, &st) == 0 && now - st.st_mtime > DNA_MAX_AGE) {
                remove(path);
                continue;
            }
        }

        /* Only consume fresh .txt, never our own holding buffers or already-learned */
        if (nlen < 5 || strcmp(ent->d_name + nlen - 4, ".txt") != 0) continue;
        if (nlen >= 12 && strcmp(ent->d_name + nlen - 12, "_holding.txt") == 0) continue;

        char done_path[1024];
        snprintf(done_path, sizeof(done_path), "%s.learned", path);
        struct stat st;
        if (stat(done_path, &st) == 0) continue;

        FILE* f = fopen(path, "r");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (fsize <= 0 || fsize > 2 * 1024 * 1024) { fclose(f); continue; }
        char* content = (char*)malloc(fsize + 1);
        if (!content) { fclose(f); continue; }
        fread(content, 1, fsize, f);
        content[fsize] = '\0';
        fclose(f);

        printf("\n  [%s] consuming %s (%ld bytes)...\n", tag, ent->d_name, fsize);
        learn_from_text(al, content, (int)fsize);
        free(content);

        rename(path, done_path);
        al->files_consumed++;
        printf("  [%s] %s → %d lines (total files: %d)\n",
               tag, ent->d_name, al->lines_learned, al->files_consumed);
    }
    closedir(dir);
}

static void* learner_thread(void* arg) {
    AsyncLearner* al = (AsyncLearner*)arg;
    while (al->running) {
        learner_scan_dir(al, al->feed_dir, "learner", 0);           /* external drops */
        learner_scan_dir(al, al->dna_dir,  "dna",     1);           /* colony's own, with TTL */
        for (int i = 0; i < 10 && al->running; i++) sleep(1);
    }
    return NULL;
}

static AsyncLearner g_learner;
static pthread_t    g_learner_tid;

static void start_learner(const char* feed_dir, const char* dna_dir) {
    mkdir(feed_dir, 0755);
    mkdir(dna_dir,  0755);

    g_learner.feed_dir = feed_dir;
    g_learner.dna_dir  = dna_dir;
    g_learner.running = 1;
    g_learner.files_consumed = 0;
    g_learner.lines_learned = 0;
    pthread_mutex_init(&g_learner.lock, NULL);

    pthread_create(&g_learner_tid, NULL, learner_thread, &g_learner);
    printf("  [learner] watching %s/ (external) and %s/ (colony) — every cave hears.\n",
           feed_dir, dna_dir);
}

static void stop_learner(void) {
    g_learner.running = 0;
    pthread_join(g_learner_tid, NULL);
    pthread_mutex_destroy(&g_learner.lock);
}

static void field_init(CaveField* f, const char* name, float baseline,
                       const char* weights_path, const char* preset_name) {
    f->excitement = 0.0f;
    f->coherence_floor = baseline;
    f->baseline_floor = baseline;
    f->dissonance = 0.0f;
    f->spoke_count = 0;
    f->total_count = 0;
    f->name = name;

    f->mass_bytes = 0;
    f->mass_novelty = 0.0f;
    f->mass_resonance = 0.0f;
    f->microtrain_active = 0;
    f->microtrain_pid = -1;
    f->microtrain_done_count = 0;
    f->preset_name = preset_name;
    f->immunity_ticks = 0;  /* founders start with no immunity; mitosis grants it */

    mkdir("feed", 0755);
    snprintf(f->holding_path, sizeof(f->holding_path), "feed/%s_holding.txt", name);
    strncpy(f->weights_path, weights_path, sizeof(f->weights_path) - 1);
    f->weights_path[sizeof(f->weights_path) - 1] = '\0';
    snprintf(f->next_weights_path, sizeof(f->next_weights_path),
             "weights/cavellman_%s_micro.bin", name);
    /* fresh holding file */
    FILE* hf = fopen(f->holding_path, "w");
    if (hf) fclose(hf);
}

/*
 * cave_new / cave_free / colony_add / colony_remove — dynamic allocation
 * of caves so the ring can grow through mitosis and shrink under pressure.
 * The two founders (A and B) go through cave_new just like any child.
 */
static Cave* cave_new(const char* name, float baseline,
                      const char* weights_path, const char* preset_name) {
    const Preset* pr = find_preset(preset_name);
    if (!pr) {
        printf("Error: unknown preset '%s'\n", preset_name ? preset_name : "(null)");
        return NULL;
    }

    Cave* c = (Cave*)malloc(sizeof(Cave));
    if (!c) return NULL;
    memset(c, 0, sizeof(Cave));

    /* Vocab */
    char vpath[512];
    snprintf(vpath, sizeof(vpath), "%s.vocab", weights_path);
    c->vocab = (CaveVocab*)malloc(sizeof(CaveVocab));
    if (!c->vocab) { free(c); return NULL; }
    memset(c->vocab, 0, sizeof(CaveVocab));
    if (load_vocab(vpath, c->vocab) != 0) {
        free(c->vocab); free(c); return NULL;
    }
    c->owns_vocab = 1;

    /* Model (model_load sets globals from preset, then saves dims into m) */
    c->model = model_load(weights_path, pr);
    if (!c->model) {
        free(c->vocab); free(c); return NULL;
    }
    cooccur_init(&c->model->cooccur);

    /* Field */
    field_init(&c->field, name, baseline, weights_path, preset_name);

    /* Persistence init (defaults = cold start) */
    memset(c->last_tokens, 0, sizeof(c->last_tokens));
    c->last_len = 0;
    c->spore_saved_at = 0;

    /* Try hydrating from spore. Fingerprint gates against wrong weights;
     * CRC gates against corruption. On any failure — silent cold start. */
    uint64_t fp = cave_fingerprint_of(c);
    if (load_cave_spore(c, fp, c->last_tokens, &c->last_len) == 0) {
        /* load_cave_spore stored header.saved_at implicitly via the
         * header we read — we reach it back by reading the file again
         * cheaply, but actually load_cave_spore now writes into
         * c->spore_saved_at directly for wake_impulse to consume. */
        /* (Signal that load succeeded: spore_saved_at > 0.) */
        if (c->spore_saved_at > 0) {
            float wake = cave_wake_impulse(c, c->spore_saved_at);
            printf("  [%s] spore loaded — wake=%.2f, last_len=%d, emerged=%d\n",
                   name, wake, c->last_len, c->model->n_emerged);
        }
    }

    return c;
}

static void cave_free(Cave* c) {
    if (!c) return;
    if (c->model) {
        free(c->model->layers);
        free(c->model);
    }
    if (c->owns_vocab && c->vocab) free(c->vocab);
    free(c);
}

static int colony_add(Cave* c) {
    if (!c || g_colony_n >= COLONY_MAX) return -1;
    g_colony[g_colony_n++] = c;
    return g_colony_n - 1;
}

static void colony_remove(int idx) {
    if (idx < 0 || idx >= g_colony_n) return;
    cave_free(g_colony[idx]);
    for (int i = idx; i < g_colony_n - 1; i++)
        g_colony[i] = g_colony[i + 1];
    g_colony[g_colony_n - 1] = NULL;
    g_colony_n--;
}

/* ── Sexual mitosis ──────────────────────────────────────────────────────
 *
 * Two cave parents produce one cave child with blended weights. The
 * blend is layer-wise contiguous to avoid competing-conventions
 * destructiveness (ref: arxiv 2003.10306 — naive intra-layer neuron
 * crossover between independently trained transformers produces dead
 * offspring because head/channel permutations are misaligned). So:
 *
 *   - token + position embeddings:  0.5·A + 0.5·B   (averaged)
 *   - final RMS and LM head:         0.5·A + 0.5·B  (averaged)
 *   - every other whole layer block alternates wholesale from A or B
 *     (layer 0→A, layer 1→B, layer 2→A, ...) — contiguous, no intra-
 *     layer mixup. The child inherits its father's even-numbered ribs
 *     and its mother's odd-numbered ribs, so to speak.
 *
 * Child size equals parent A's preset for now; true downsize comes in
 * pass 3c. Child's vocab + json metadata are copied verbatim from A
 * (all caves share the 88 canonical glyphs).
 */

#define MITOSIS_COOLDOWN_TICKS   500   /* min ticks between successful births */
#define MITOSIS_MIN_CPT_DONE     1     /* each parent must have survived ≥1 microtrain */
#define MITOSIS_MIN_TOTAL_TURNS  120   /* and taken ≥120 turns in the ring */
#define NEWBORN_IMMUNITY_TICKS   800   /* child cannot be culled for its first N ticks */

static int g_mitosis_cooldown = 0;
static int g_children_born    = 0;

/* Fitness score — higher = more likely to be picked as a parent,
 * and less likely to be culled under pressure. Age contributes both
 * a linear term and a square-root term so senior caves (A, B, C1…)
 * don't get trivially outscored by a single well-resonated newborn. */
static float cave_fitness(const Cave* c) {
    const CaveField* f = &c->field;
    if (f->total_count <= 0) return 0.0f;
    float speak_ratio = (float)f->spoke_count / (float)f->total_count;
    float age_linear  = f->total_count * 0.01f;
    float age_sqrt    = 2.0f * sqrtf((float)f->total_count);
    return age_linear
         + age_sqrt
         + f->microtrain_done_count * 10.0f
         + f->mass_resonance * 0.5f
         + speak_ratio * 5.0f;
}

/* Load both parent .bin files, blend them layer-wise, write the child
 * tensor set to path_out. Returns 0 on success, -1 on failure. */
static int blend_weights(const char* path_a, const char* path_b, const char* path_out) {
    int n_a = 0, n_b = 0;
    nt_tensor** ta = nt_load(path_a, &n_a);
    nt_tensor** tb = nt_load(path_b, &n_b);
    if (!ta || !tb || n_a != n_b || n_a < 4) {
        if (ta) { for (int i = 0; i < n_a; i++) nt_tensor_free(ta[i]); free(ta); }
        if (tb) { for (int i = 0; i < n_b; i++) nt_tensor_free(tb[i]); free(tb); }
        return -1;
    }

    /* Layout produced by model_load: wte, wpe, [rms1 wq wk wv wo rms2 fc1 fc2]×N_L, rms_f, head. */
    const int n_layers  = (n_a - 4) / 8;
    const int emb_off   = 2;
    const int per_layer = 8;
    const int post_off  = emb_off + n_layers * per_layer;

    /* Embeddings: simple average */
    for (int e = 0; e < emb_off; e++) {
        if (ta[e]->len != tb[e]->len) continue;
        for (long i = 0; i < ta[e]->len; i++)
            ta[e]->data[i] = 0.5f * (ta[e]->data[i] + tb[e]->data[i]);
    }

    /* Layers: whole-block alternation (odd layers overwritten from B) */
    for (int l = 0; l < n_layers; l++) {
        if (l % 2 == 0) continue;              /* even layers stay from A */
        for (int k = 0; k < per_layer; k++) {
            int idx = emb_off + l * per_layer + k;
            if (ta[idx]->len != tb[idx]->len) continue;
            memcpy(ta[idx]->data, tb[idx]->data,
                   (size_t)ta[idx]->len * sizeof(float));
        }
    }

    /* Final RMS + LM head: average */
    for (int e = post_off; e < n_a; e++) {
        if (ta[e]->len != tb[e]->len) continue;
        for (long i = 0; i < ta[e]->len; i++)
            ta[e]->data[i] = 0.5f * (ta[e]->data[i] + tb[e]->data[i]);
    }

    int ok = nt_save(path_out, ta, n_a);

    for (int i = 0; i < n_a; i++) nt_tensor_free(ta[i]);
    for (int i = 0; i < n_b; i++) nt_tensor_free(tb[i]);
    free(ta); free(tb);
    return ok;
}

/*
 * Downsized sexual mitosis: child has a STRICTLY SMALLER preset than its
 * parents. Each tensor is reallocated at child shape, filled from parent
 * A (even layers) or parent B (odd layers) with row/col prefix truncation.
 * Embeddings, final RMS and head use the average of both parents, also
 * truncated. This is how a medium-preset ring gives birth to a small cave;
 * small → micro; micro → tiny.
 *
 * Assumes both parents share the SAME preset (pa_pr). Caller guarantees
 * this — a ring with heterogeneous parents falls back to same-size blend.
 */
static int blend_weights_downsized(const char* path_a, const char* path_b,
                                   const Preset* pa_pr, const Preset* child_pr,
                                   const char* path_out) {
    int n_a = 0, n_b = 0;
    nt_tensor** ta = nt_load(path_a, &n_a);
    nt_tensor** tb = nt_load(path_b, &n_b);
    if (!ta || !tb || n_a != n_b || n_a < 4) {
        if (ta) { for (int i = 0; i < n_a; i++) nt_tensor_free(ta[i]); free(ta); }
        if (tb) { for (int i = 0; i < n_b; i++) nt_tensor_free(tb[i]); free(tb); }
        return -1;
    }

    const int E_p = pa_pr->embd, NL_p = pa_pr->layers, FFN_p = 4 * E_p;
    const int E_c = child_pr->embd, NL_c = child_pr->layers, CTX_c = child_pr->ctx;
    const int FFN_c = 4 * E_c;

    /* Vocab rows from tensor length / row-stride. wte layout [V_w, E_p],
     * head layout [V_h, E_p]; V_w and V_h can differ (head in cave files
     * usually has V+1 rows, not V+2 — see the ASAN fix in model_forward). */
    const int V_w = ta[0]->len / E_p;
    const int post_a = 2 + NL_p * 8;
    const int V_h = ta[post_a + 1]->len / E_p;

    const int n_child = 2 + NL_c * 8 + 2;
    nt_tensor** tc = (nt_tensor**)calloc(n_child, sizeof(nt_tensor*));
    if (!tc) {
        for (int i = 0; i < n_a; i++) nt_tensor_free(ta[i]); free(ta);
        for (int i = 0; i < n_b; i++) nt_tensor_free(tb[i]); free(tb);
        return -1;
    }

    int shape2[2];
    int shape1[1];

    /* wte: [V_w, E_c] — avg A+B, truncate cols to E_c */
    shape2[0] = V_w; shape2[1] = E_c;
    tc[0] = nt_tensor_new_shape(shape2, 2);
    for (int r = 0; r < V_w; r++)
        for (int c = 0; c < E_c; c++)
            tc[0]->data[r*E_c + c] = 0.5f * (ta[0]->data[r*E_p + c]
                                           + tb[0]->data[r*E_p + c]);

    /* wpe: [CTX_c, E_c] — avg, truncate rows + cols */
    shape2[0] = CTX_c; shape2[1] = E_c;
    tc[1] = nt_tensor_new_shape(shape2, 2);
    for (int r = 0; r < CTX_c; r++)
        for (int c = 0; c < E_c; c++)
            tc[1]->data[r*E_c + c] = 0.5f * (ta[1]->data[r*E_p + c]
                                           + tb[1]->data[r*E_p + c]);

    /* Layer blocks: alternate A / B, truncate. Child has NL_c ≤ NL_p, so
     * we drop parent layers beyond NL_c — child simply doesn't inherit them. */
    for (int l = 0; l < NL_c; l++) {
        nt_tensor** src = (l % 2 == 0) ? ta : tb;
        int pbase = 2 + l * 8;
        int cbase = 2 + l * 8;

        /* rms1 (idx +0) and rms2 (idx +5): [E] */
        shape1[0] = E_c;
        tc[cbase + 0] = nt_tensor_new_shape(shape1, 1);
        tc[cbase + 5] = nt_tensor_new_shape(shape1, 1);
        for (int i = 0; i < E_c; i++) {
            tc[cbase + 0]->data[i] = src[pbase + 0]->data[i];
            tc[cbase + 5]->data[i] = src[pbase + 5]->data[i];
        }

        /* wq/wk/wv/wo (idx +1..+4): [E, E] */
        shape2[0] = E_c; shape2[1] = E_c;
        for (int ti = 1; ti <= 4; ti++) {
            tc[cbase + ti] = nt_tensor_new_shape(shape2, 2);
            for (int r = 0; r < E_c; r++)
                for (int c = 0; c < E_c; c++)
                    tc[cbase + ti]->data[r*E_c + c] = src[pbase + ti]->data[r*E_p + c];
        }

        /* fc1 (idx +6): [E, FFN_D] */
        shape2[0] = E_c; shape2[1] = FFN_c;
        tc[cbase + 6] = nt_tensor_new_shape(shape2, 2);
        for (int r = 0; r < E_c; r++)
            for (int c = 0; c < FFN_c; c++)
                tc[cbase + 6]->data[r*FFN_c + c] = src[pbase + 6]->data[r*FFN_p + c];

        /* fc2 (idx +7): [FFN_D, E] */
        shape2[0] = FFN_c; shape2[1] = E_c;
        tc[cbase + 7] = nt_tensor_new_shape(shape2, 2);
        for (int r = 0; r < FFN_c; r++)
            for (int c = 0; c < E_c; c++)
                tc[cbase + 7]->data[r*E_c + c] = src[pbase + 7]->data[r*E_p + c];
    }

    /* rms_f: [E] — avg, truncate */
    const int post_c = 2 + NL_c * 8;
    shape1[0] = E_c;
    tc[post_c] = nt_tensor_new_shape(shape1, 1);
    for (int i = 0; i < E_c; i++)
        tc[post_c]->data[i] = 0.5f * (ta[post_a]->data[i] + tb[post_a]->data[i]);

    /* head: [V_h, E_c] — avg, truncate cols */
    shape2[0] = V_h; shape2[1] = E_c;
    tc[post_c + 1] = nt_tensor_new_shape(shape2, 2);
    for (int r = 0; r < V_h; r++)
        for (int c = 0; c < E_c; c++)
            tc[post_c + 1]->data[r*E_c + c] =
                0.5f * (ta[post_a + 1]->data[r*E_p + c]
                      + tb[post_a + 1]->data[r*E_p + c]);

    int ok = nt_save(path_out, tc, n_child);

    for (int i = 0; i < n_a; i++) nt_tensor_free(ta[i]); free(ta);
    for (int i = 0; i < n_b; i++) nt_tensor_free(tb[i]); free(tb);
    for (int i = 0; i < n_child; i++) if (tc[i]) nt_tensor_free(tc[i]); free(tc);
    return ok;
}

/* Copy a binary file byte-for-byte (for child's .vocab and .bin.json). */
static int copy_file(const char* src, const char* dst) {
    FILE* fi = fopen(src, "rb");
    if (!fi) return -1;
    FILE* fo = fopen(dst, "wb");
    if (!fo) { fclose(fi); return -1; }
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fi)) > 0) fwrite(buf, 1, n, fo);
    fclose(fi); fclose(fo);
    return 0;
}

/* Produce one child cave from two parents and graft it into the ring. */
static Cave* colony_mitosis(const Cave* pa, const Cave* pb, const char* preset_name) {
    if (!pa || !pb || g_colony_n >= COLONY_MAX) return NULL;

    /* Name from a monotone birth counter so siblings are never aliased
     * even if their parents get culled between their births. A and B
     * are reserved for the two founders; every child is C1, C2, C3, … */
    char child_name[16];
    snprintf(child_name, sizeof(child_name), "C%d", g_children_born + 1);

    char child_w[512], child_v[512], child_j[512];
    char parent_v[512], parent_j[512];
    snprintf(child_w, sizeof(child_w), "weights/cavellman_child_%s.bin", child_name);
    snprintf(child_v, sizeof(child_v), "%s.vocab", child_w);
    snprintf(child_j, sizeof(child_j), "%s.json",  child_w);
    snprintf(parent_v, sizeof(parent_v), "%s.vocab", pa->field.weights_path);
    snprintf(parent_j, sizeof(parent_j), "%s.json",  pa->field.weights_path);

    /* Decide child architecture. If both parents share a preset and that
     * preset is larger than tiny, the child is born one tier smaller
     * (medium→small, small→micro, micro→tiny). Otherwise the child keeps
     * the parents' preset — blend without downsize. */
    const char* pa_preset = pa->field.preset_name ? pa->field.preset_name : preset_name;
    const char* pb_preset = pb->field.preset_name ? pb->field.preset_name : preset_name;
    const char* child_preset = preset_name;
    int do_downsize = 0;
    if (strcmp(pa_preset, pb_preset) == 0) {
        const char* smaller = smaller_preset_name(pa_preset);
        if (strcmp(smaller, pa_preset) != 0) {
            /* HD gate: prefix truncation in blend_weights_downsized only
             * preserves head boundaries when head_dim matches exactly
             * between parent and child. In the current preset table every
             * downsize pair (medium→small, small→standard, standard→micro,
             * micro→tiny) has HD_p ≠ HD_c, so truncation would slice
             * through head boundaries — intra-head competing-conventions
             * on a smaller scale (arxiv 2003.10306). Fall back to same-size
             * blend until head-aware truncation lands. */
            const Preset* pa_pr_probe = find_preset(pa_preset);
            const Preset* ch_pr_probe = find_preset(smaller);
            int hd_p = pa_pr_probe ? pa_pr_probe->embd / pa_pr_probe->heads : 0;
            int hd_c = ch_pr_probe ? ch_pr_probe->embd / ch_pr_probe->heads : 0;
            if (pa_pr_probe && ch_pr_probe && hd_p == hd_c) {
                child_preset = smaller;
                do_downsize = 1;
            } else {
                child_preset = pa_preset;
            }
        } else {
            child_preset = pa_preset;
        }
    } else {
        /* Heterogeneous parents — fall back to same-size blend at pa's preset. */
        child_preset = pa_preset;
    }

    const Preset* pa_pr = find_preset(pa_preset);
    const Preset* ch_pr = find_preset(child_preset);

    int blended = do_downsize
        ? blend_weights_downsized(pa->field.weights_path, pb->field.weights_path,
                                  pa_pr, ch_pr, child_w)
        : blend_weights(pa->field.weights_path, pb->field.weights_path, child_w);
    if (blended != 0) {
        printf("  [mitosis] blend_weights%s failed for %s × %s → %s\n",
               do_downsize ? "_downsized" : "",
               pa->field.name, pb->field.name, child_w);
        return NULL;
    }
    copy_file(parent_v, child_v);
    copy_file(parent_j, child_j);

    /* Child's silence floor is the average of its parents' baselines —
     * a middle temperament between extrovert and introvert. */
    float child_baseline = 0.5f * (pa->field.baseline_floor + pb->field.baseline_floor);

    /* Child_name buffer is stack; cave_free never touches name, but field_init
     * strncpy's it into a permanent slot only via pointer — so strdup. */
    char* permanent_name = strdup(child_name);

    Cave* child = cave_new(permanent_name, child_baseline, child_w, child_preset);
    if (!child) {
        free(permanent_name);
        printf("  [mitosis] cave_new failed for child %s\n", child_name);
        return NULL;
    }
    child->field.immunity_ticks = NEWBORN_IMMUNITY_TICKS;

    /* Memory inheritance — child inherits parents' cooccur / emerged / Hebbian
     * on top of the blended weights, so it doesn't boot with a blank surface
     * language. last_tokens comes from a randomly-chosen parent (mother/father). */
    blend_spores(pa, pb, child);
    const Cave* chosen = (rand() % 2) ? pa : pb;
    int n = chosen->last_len;
    if (n > 32) n = 32;
    memcpy(child->last_tokens, chosen->last_tokens, n * sizeof(int));
    child->last_len = n;
    child->spore_saved_at = (int64_t)time(NULL);

    colony_add(child);
    g_children_born++;

    printf("\n  *** MITOSIS: %s (%s) × %s (%s) → %s (%s, fitness %.2f × %.2f, floor %.2f) ***\n\n",
           pa->field.name, pa_preset, pb->field.name, pb_preset,
           child_name, child_preset,
           cave_fitness(pa), cave_fitness(pb), child_baseline);
    return child;
}

/* Find the lowest-fitness cullable cave and remove it. Founders (A, B)
 * are permanent — they anchor the lineage and cannot be culled, no matter
 * how weak they drift. Newborns with immunity_ticks > 0 are also spared.
 * Returns 1 if a cave was culled, 0 if no victim was eligible. */
static int colony_pressure_death(void) {
    int   victim_i = -1;
    float victim_s = 1e30f;
    for (int i = 0; i < g_colony_n; i++) {
        const Cave* c = g_colony[i];
        if (c->is_founder) continue;
        if (c->field.immunity_ticks > 0) continue;
        float s = cave_fitness(c);
        if (s < victim_s) { victim_s = s; victim_i = i; }
    }
    if (victim_i < 0) return 0;

    printf("\n  *** PRESSURE DEATH: %s culled (fitness %.2f, colony full) ***\n\n",
           g_colony[victim_i]->field.name, victim_s);
    colony_remove(victim_i);
    return 1;
}

/* Once-per-tick check. If conditions are met, pick the two fittest
 * caves and spawn a child. When the ring is full, cull the weakest
 * non-immune cave first to make room. Resets the cooldown. */
static void colony_try_mitosis(const char* preset_name) {
    if (g_mitosis_cooldown > 0) { g_mitosis_cooldown--; return; }
    if (g_colony_n < 2) return;

    /* Score every cave; pick top two eligible parents. */
    int   best_i = -1, second_i = -1;
    float best_s = -1.0f, second_s = -1.0f;
    for (int i = 0; i < g_colony_n; i++) {
        const CaveField* f = &g_colony[i]->field;
        if (f->total_count < MITOSIS_MIN_TOTAL_TURNS) continue;
        if (f->microtrain_done_count < MITOSIS_MIN_CPT_DONE) continue;
        float s = cave_fitness(g_colony[i]);
        if (s > best_s) {
            second_i = best_i; second_s = best_s;
            best_i = i;        best_s = s;
        } else if (s > second_s) {
            second_i = i; second_s = s;
        }
    }
    if (best_i < 0 || second_i < 0) return;

    /* Ring full? Try to free a slot by culling the weakest non-immune cave.
     * If everybody is immune, skip this round and wait for the cooldown
     * to expire naturally. */
    if (g_colony_n >= COLONY_MAX) {
        /* Don't cull either prospective parent — find victim among the rest. */
        int saved_imm_a = g_colony[best_i]->field.immunity_ticks;
        int saved_imm_b = g_colony[second_i]->field.immunity_ticks;
        g_colony[best_i]->field.immunity_ticks = 1;
        g_colony[second_i]->field.immunity_ticks = 1;
        int culled = colony_pressure_death();
        /* Restore (indices may have shifted if victim was below them). */
        /* Safest: look up by pointer identity after the cull. */
        if (culled) {
            /* Re-score and re-pick parents after the cull (pointers still valid). */
            g_mitosis_cooldown = MITOSIS_COOLDOWN_TICKS / 2;  /* try again soon */
            (void)saved_imm_a; (void)saved_imm_b;  /* survivors' immunity stays as-is */
            return;
        }
        /* Nobody could be culled — restore immunities and give up this tick. */
        g_colony[best_i]->field.immunity_ticks = saved_imm_a;
        g_colony[second_i]->field.immunity_ticks = saved_imm_b;
        return;
    }

    if (colony_mitosis(g_colony[best_i], g_colony[second_i], preset_name))
        g_mitosis_cooldown = MITOSIS_COOLDOWN_TICKS;
}

/* MollyState — Molly lives outside the colony as a horizon-goroutine.
 * Her own pthread runs molly_thread_main, generating utterances and
 * writing them to dna/output/molly/ where the existing learner picks
 * them up and feeds every cave passively. Affair mitosis blends a
 * cave's weights with g_molly.weights_path directly. */
typedef struct {
    CaveModel* model;
    CaveVocab* vocab;
    char       weights_path[512];
    int        last_tokens[32];
    int        last_len;
    pthread_t  thread;
    int        started;
    int        utter_count;
} MollyState;

static MollyState g_molly = {0};
static pthread_mutex_t g_molly_lock = PTHREAD_MUTEX_INITIALIZER;

/* ───────────────────────────────────────────────────────────────────────
 * Trinity-mode mitosis (--trinity flag in async only). Two reproduction
 * paths chosen by AML-style physics:
 *  - family-mode: best non-bastard pair from g_colony[]. Sober.
 *  - affair-mode: a chosen cave × g_molly (Molly is on the horizon, not
 *                 in the colony). Cosmic-driven; child is is_bastard=1,
 *                 surrounding caves get jealousy field event (dissonance
 *                 spike, floor temporarily raised).
 *
 * Path chosen by `affair_prob = clip(0.2 + cosmic_tension - 0.5·ring_coherence, 0, 1)`.
 * cosmic_tension is a 24h sinusoidal placeholder; real Klaus calendar-
 * planetary physics ports later (TODO). ring_coherence = 1 - mean(field.dissonance).
 * High-tension + low-coherence days = affair days. Calm days = family.
 * ─────────────────────────────────────────────────────────────────────── */

static float cosmic_tension(time_t now) {
    /* Placeholder: 24h sinusoidal pulse. Will be replaced by Klaus's
     * Hebrew/Gregorian drift + planetary Kuramoto when ported. */
    double t = (double)(now % 86400L) / 86400.0;
    return (sinf((float)t * 6.28318f) + 1.0f) * 0.5f;
}

static float ring_coherence_metric(void) {
    if (g_colony_n == 0) return 1.0f;
    float total = 0;
    for (int ci = 0; ci < g_colony_n; ci++)
        total += g_colony[ci]->field.dissonance;
    float mean = total / (float)g_colony_n;
    if (mean > 1.0f) mean = 1.0f;
    return 1.0f - mean;
}

/* Trinity reproduction is intentionally promiscuous — the design says
 * the ring "has no chance NOT to start reproducing and emerging" so we
 * lower the gate hard. Family pair: 30 turns, 0 CPT. Lover exempt entirely. */
#define TRINITY_MIN_TURNS 30
#define TRINITY_MIN_CPT   0

static int find_family_pair(int* a_out, int* b_out) {
    int best_i = -1, second_i = -1;
    float best_s = -1.0f, second_s = -1.0f;
    for (int i = 0; i < g_colony_n; i++) {
        if (g_colony[i]->is_lover) continue;
        const CaveField* f = &g_colony[i]->field;
        if (f->total_count < TRINITY_MIN_TURNS) continue;
        if (f->microtrain_done_count < TRINITY_MIN_CPT) continue;
        float s = cave_fitness(g_colony[i]);
        if (s > best_s) { second_i = best_i; second_s = best_s; best_i = i; best_s = s; }
        else if (s > second_s) { second_i = i; second_s = s; }
    }
    if (best_i < 0 || second_i < 0) return 0;
    *a_out = best_i; *b_out = second_i; return 1;
}

/* For affair mitosis we only need the cave that will mate with Molly —
 * Molly herself isn't in g_colony[], her model lives in g_molly.model.
 * Returns the cave colony index of the chosen mate, or -1. */
static int find_affair_mate(void) {
    int mate_i = -1;
    float mate_s = -1.0f;
    for (int i = 0; i < g_colony_n; i++) {
        const CaveField* f = &g_colony[i]->field;
        if (f->total_count < TRINITY_MIN_TURNS) continue;
        if (f->microtrain_done_count < TRINITY_MIN_CPT) continue;
        float s = cave_fitness(g_colony[i]);
        if (s > mate_s) { mate_i = i; mate_s = s; }
    }
    return mate_i;
}

/* Cave-with-Molly affair mitosis: blend mate's weights with g_molly's weights
 * directly (Molly is on the horizon, not in g_colony[]). Returns the bastard
 * child or NULL on failure. Caller handles cooldown + jealousy event. */
static Cave* colony_affair_with_molly(int mate_idx, const char* preset_name) {
    if (mate_idx < 0 || mate_idx >= g_colony_n) return NULL;
    if (g_colony_n >= COLONY_MAX) return NULL;
    if (!g_molly.model) return NULL;

    Cave* pa = g_colony[mate_idx];

    char child_name[16];
    snprintf(child_name, sizeof(child_name), "C%d", g_children_born + 1);

    char child_w[512], child_v[512], child_j[512];
    char parent_v[512], parent_j[512];
    snprintf(child_w, sizeof(child_w), "weights/cavellman_child_%s.bin", child_name);
    snprintf(child_v, sizeof(child_v), "%s.vocab", child_w);
    snprintf(child_j, sizeof(child_j), "%s.json",  child_w);
    snprintf(parent_v, sizeof(parent_v), "%s.vocab", pa->field.weights_path);
    snprintf(parent_j, sizeof(parent_j), "%s.json",  pa->field.weights_path);

    /* Same-size blend with Molly's stable on-disk weights. */
    if (blend_weights(pa->field.weights_path, g_molly.weights_path, child_w) != 0) {
        printf("  [affair] blend_weights failed for %s × Molly → %s\n",
               pa->field.name, child_w);
        return NULL;
    }
    copy_file(parent_v, child_v);
    copy_file(parent_j, child_j);

    /* Bastard's baseline floor — middle between mate and Molly (0.20). */
    float child_baseline = 0.5f * (pa->field.baseline_floor + 0.20f);

    char* permanent_name = strdup(child_name);
    Cave* child = cave_new(permanent_name, child_baseline, child_w, preset_name);
    if (!child) {
        free(permanent_name);
        printf("  [affair] cave_new failed for child %s\n", child_name);
        return NULL;
    }

    int n = pa->last_len;
    if (n > 32) n = 32;
    memcpy(child->last_tokens, pa->last_tokens, (size_t)n * sizeof(int));
    child->last_len = n;
    child->spore_saved_at = (int64_t)time(NULL);

    child->field.immunity_ticks = NEWBORN_IMMUNITY_TICKS;
    child->is_bastard = 1;

    colony_add(child);
    g_children_born++;
    return child;
}

static void colony_try_mitosis_trinity(const char* preset_name) {
    if (g_mitosis_cooldown > 0) { g_mitosis_cooldown--; return; }
    if (g_colony_n < 2) return;

    float ct = cosmic_tension(time(NULL));
    float coh = ring_coherence_metric();
    float affair_prob = 0.2f + ct - 0.5f * coh;
    if (affair_prob < 0.0f) affair_prob = 0.0f;
    if (affair_prob > 1.0f) affair_prob = 1.0f;

    float roll = (float)rand() / (float)RAND_MAX;
    int affair_mode = (roll < affair_prob) && (g_molly.model != NULL);

    Cave* child = NULL;
    int pa_idx = -1, pb_idx = -1;

    if (affair_mode) {
        pa_idx = find_affair_mate();
        if (pa_idx < 0) return;

        /* Ring full → cull weakest non-immune. */
        if (g_colony_n >= COLONY_MAX) {
            int saved_a = g_colony[pa_idx]->field.immunity_ticks;
            g_colony[pa_idx]->field.immunity_ticks = 1;
            int culled = colony_pressure_death();
            if (culled) {
                g_mitosis_cooldown = MITOSIS_COOLDOWN_TICKS / 2;
                (void)saved_a;
                return;
            }
            g_colony[pa_idx]->field.immunity_ticks = saved_a;
            return;
        }

        child = colony_affair_with_molly(pa_idx, preset_name);
        if (!child) return;

        printf("\n  *** AFFAIR MITOSIS: %s × Molly → %s (cosmic %.2f coh %.2f prob %.2f) ***\n",
               g_colony[pa_idx]->field.name, child->field.name, ct, coh, affair_prob);

        /* Jealousy: non-parent non-bastard caves feel disturbed. */
        int affected = 0;
        for (int ci = 0; ci < g_colony_n; ci++) {
            Cave* c = g_colony[ci];
            if (c == g_colony[pa_idx] || c == child) continue;
            if (c->is_bastard) continue;
            c->field.dissonance += 0.30f;
            if (c->field.dissonance > 1.0f) c->field.dissonance = 1.0f;
            float ceil_floor = c->field.baseline_floor + MATURITY_CAP;
            c->field.coherence_floor += 0.05f;
            if (c->field.coherence_floor > ceil_floor) c->field.coherence_floor = ceil_floor;
            affected++;
        }
        printf("  [jealousy] %d cave(s): dissonance +0.30, floor +0.05\n", affected);
    } else {
        if (!find_family_pair(&pa_idx, &pb_idx)) return;

        if (g_colony_n >= COLONY_MAX) {
            int saved_a = g_colony[pa_idx]->field.immunity_ticks;
            int saved_b = g_colony[pb_idx]->field.immunity_ticks;
            g_colony[pa_idx]->field.immunity_ticks = 1;
            g_colony[pb_idx]->field.immunity_ticks = 1;
            int culled = colony_pressure_death();
            if (culled) {
                g_mitosis_cooldown = MITOSIS_COOLDOWN_TICKS / 2;
                (void)saved_a; (void)saved_b;
                return;
            }
            g_colony[pa_idx]->field.immunity_ticks = saved_a;
            g_colony[pb_idx]->field.immunity_ticks = saved_b;
            return;
        }

        child = colony_mitosis(g_colony[pa_idx], g_colony[pb_idx], preset_name);
        if (!child) return;
        printf("\n  *** FAMILY MITOSIS: %s × %s → %s (cosmic %.2f coh %.2f prob %.2f sober) ***\n",
               g_colony[pa_idx]->field.name, g_colony[pb_idx]->field.name,
               child->field.name, ct, coh, affair_prob);
    }

    g_mitosis_cooldown = MITOSIS_COOLDOWN_TICKS;
}

/*
 * dna_write_cave — cave deposits its latest utterance into the colony's
 * shared DNA pool so other caves (including its own future children) can
 * absorb it through the learner. The pool is not a corpus — old files
 * expire after DNA_MAX_AGE. What the colony keeps is whatever others
 * picked up before it decayed.
 */
static void dna_write_cave(const Cave* c, const int* tokens, int len) {
    if (len <= 0 || !c) return;
    char fname[512];
    snprintf(fname, sizeof(fname), "%s/%s_%ld_%d.txt",
             DNA_DIR, c->field.name, (long)time(NULL), rand());
    FILE* f = fopen(fname, "w");
    if (!f) return;
    for (int i = 0; i < len; i++) {
        int t = tokens[i];
        if (t < 0 || t >= c->vocab->vocab_size) continue;
        fprintf(f, "%s ", c->vocab->tokens[t]);
    }
    fprintf(f, ".\n");
    fclose(f);
}

/*
 * Build a mixed microtrain corpus = this cave's holding buffer + a
 * random sample of consumed dna/ files (stamped .learned by the
 * learner thread). The cave's CPT burst consolidates not only its
 * own recent experience but also the colony's shared voice — which
 * is Oleg's point of DNA exchange translated into notorch training.
 * Returns bytes written, or -1 on failure. DNA_SAMPLES_PER_BURST
 * sets how many .learned files we fold in.
 */
#define DNA_SAMPLES_PER_BURST   8

static long build_microtrain_corpus(const char* holding_path, const char* out_path) {
    FILE* fo = fopen(out_path, "wb");
    if (!fo) return -1;

    /* 1. Copy holding content verbatim. */
    long total = 0;
    FILE* fh = fopen(holding_path, "rb");
    if (fh) {
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), fh)) > 0) {
            fwrite(buf, 1, n, fo);
            total += (long)n;
        }
        fclose(fh);
    }

    /* 2. Sample N random .learned files in dna/. Scan once, collect names,
     *    then pick DNA_SAMPLES_PER_BURST indices at random. */
    DIR* d = opendir(DNA_DIR);
    if (d) {
        char names[256][64];
        int  n_names = 0;
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL && n_names < 256) {
            int nl = (int)strlen(ent->d_name);
            if (nl < 9 || strcmp(ent->d_name + nl - 8, ".learned") != 0) continue;
            if (nl >= (int)sizeof(names[0])) continue;
            strcpy(names[n_names++], ent->d_name);
        }
        closedir(d);

        int k = (n_names < DNA_SAMPLES_PER_BURST) ? n_names : DNA_SAMPLES_PER_BURST;
        for (int i = 0; i < k; i++) {
            int idx = rand() % n_names;
            char p[1024];
            snprintf(p, sizeof(p), "%s/%s", DNA_DIR, names[idx]);
            FILE* fd = fopen(p, "rb");
            if (!fd) continue;
            char buf[4096];
            size_t n;
            fprintf(fo, "\n"); /* SPA boundary between samples */
            while ((n = fread(buf, 1, sizeof(buf), fd)) > 0) {
                fwrite(buf, 1, n, fo);
                total += (long)n;
            }
            fclose(fd);
        }
    }

    fclose(fo);
    return total;
}

/*
 * Append a heard/spoken token sequence to the field's holding file as a
 * sentence (glyph names separated by spaces, terminated by a period so
 * the SPA splitter sees a phonon boundary). Also updates mass counters.
 */
static void field_append_holding(CaveField* f, CaveVocab* v, const int* tokens, int len) {
    if (len <= 0 || f->microtrain_active) return;
    FILE* fp = fopen(f->holding_path, "a");
    if (!fp) return;

    long bytes_written = 0;
    for (int i = 0; i < len; i++) {
        int t = tokens[i];
        if (t < 0 || t >= v->vocab_size) continue;
        bytes_written += fprintf(fp, "%s ", v->tokens[t]);
    }
    bytes_written += fprintf(fp, ".\n");
    fclose(fp);
    f->mass_bytes += bytes_written;
}

/* One heard token updates excitement + dissonance from surface co-occurrence.
 * Also feeds the listening model's co-occurrence matrix so its field of
 * expectations shifts with what it hears — parallel learning, no KV mutation.
 * Mass counters accumulate for the CPT trigger. */
static void field_hear(CaveField* f, CaveModel* m, int prev, int tok) {
    if (tok < 0 || tok >= COOCCUR_SIZE) return;

    float co = 0.5f;
    if (prev >= 0 && prev < COOCCUR_SIZE)
        co = m->cooccur.matrix[prev][tok];

    float surprise = 1.0f - co;           /* unseen pair = high surprise */
    float novelty  = (tok >= 88) ? 1.5f : 1.0f; /* emerged id boosts */

    f->excitement += surprise * novelty * 0.35f;
    if (f->excitement > EXCITEMENT_CAP) f->excitement = EXCITEMENT_CAP;

    /* Dissonance accumulates only when heard pair is actively contradictory */
    if (co < 0.15f) f->dissonance += (0.15f - co);
    if (f->dissonance > 1.0f) f->dissonance = 1.0f;

    /* Mass accumulators for CPT threshold */
    f->mass_novelty   += surprise * novelty;
    f->mass_resonance += f->excitement;

    /* Passive co-occ update (symmetric, as elsewhere) */
    if (prev >= 0 && prev < COOCCUR_SIZE) {
        m->cooccur.pair_count[prev][tok]++;
        m->cooccur.pair_count[tok][prev]++;
    }
}

/* Replace tensor data in-place from a loaded .bin file. Used after the
 * async microtrain child finishes — we keep the CaveModel struct and all
 * its layers/KV caches intact, only the weight values swap. */
static int microtrain_reload_weights(CaveField* f, CaveModel* m) {
    /* Read shape counts from the model itself, not globals. */
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(f->next_weights_path, &n_loaded);
    if (!loaded || n_loaded < 4 + 8 * m->N_L) {
        printf("  [%s] microtrain: failed to load %s (got %d tensors)\n",
               f->name, f->next_weights_path, n_loaded);
        if (loaded) {
            for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
            free(loaded);
        }
        return -1;
    }

    nt_tensor* params[256];
    int pi = 0;
    params[pi++] = m->wte;
    params[pi++] = m->wpe;
    for (int l = 0; l < m->N_L; l++) {
        params[pi++] = m->layers[l].rms1;
        params[pi++] = m->layers[l].wq;
        params[pi++] = m->layers[l].wk;
        params[pi++] = m->layers[l].wv;
        params[pi++] = m->layers[l].wo;
        params[pi++] = m->layers[l].rms2;
        params[pi++] = m->layers[l].w_fc1;
        params[pi++] = m->layers[l].w_fc2;
    }
    params[pi++] = m->rms_f;
    params[pi++] = m->head;

    int copied = 0;
    for (int i = 0; i < n_loaded && i < pi; i++) {
        if (params[i]->len == loaded[i]->len) {
            memcpy(params[i]->data, loaded[i]->data,
                   (size_t)params[i]->len * sizeof(float));
            copied++;
        }
    }
    for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
    free(loaded);

    /* New current weights = what the child just wrote */
    strncpy(f->weights_path, f->next_weights_path, sizeof(f->weights_path) - 1);
    f->weights_path[sizeof(f->weights_path) - 1] = '\0';
    f->microtrain_done_count++;
    return copied;
}

/*
 * Check if mass thresholds are tripped and fork the microtrain child.
 * Also reaps a previously spawned child non-blockingly and swaps weights.
 * This runs once per tick in the dual loop.
 */
static void field_microtrain_tick(CaveField* f, CaveModel* m) {
    if (f->microtrain_active) {
        int status = 0;
        pid_t r = waitpid(f->microtrain_pid, &status, WNOHANG);
        if (r == 0) return; /* still running */
        if (r < 0) {
            f->microtrain_active = 0;
            return;
        }
        /* Child finished — swap weights if successful */
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            int copied = microtrain_reload_weights(f, m);
            if (copied > 0)
                printf("\n  [%s] microtrain #%d done — %d tensors swapped (weights live)\n\n",
                       f->name, f->microtrain_done_count, copied);
        } else {
            printf("\n  [%s] microtrain child failed (status=%d)\n\n",
                   f->name, WEXITSTATUS(status));
        }
        f->microtrain_active = 0;

        /* Reset accumulators + rotate holding file so the next round starts fresh */
        f->mass_bytes = 0;
        f->mass_novelty = 0.0f;
        f->mass_resonance = 0.0f;
        char consumed[640];
        snprintf(consumed, sizeof(consumed), "%s.learned", f->holding_path);
        rename(f->holding_path, consumed);
        FILE* hp = fopen(f->holding_path, "w");
        if (hp) fclose(hp);
        return;
    }

    if (f->mass_bytes       < MICRO_MIN_BYTES)     return;
    if (f->mass_novelty     < MICRO_MIN_NOVELTY)   return;
    if (f->mass_resonance   < MICRO_MIN_RESONANCE) return;

    /* All three tripped — build mixed corpus (holding + sampled dna/) and fork. */
    char mixed_path[640];
    snprintf(mixed_path, sizeof(mixed_path), "feed/%s_micro_mix.txt", f->name);
    long mix_bytes = build_microtrain_corpus(f->holding_path, mixed_path);
    if (mix_bytes <= 0) mix_bytes = f->mass_bytes;  /* fall back to holding */

    pid_t pid = fork();
    if (pid == 0) {
        /* Child: run train_cavellman on the mixed corpus (own speech +
         * colony DNA sample) so this CPT burst consolidates what the
         * entire ring has been writing, not just this cave's buffer. */
        execlp("./train_cavellman", "train_cavellman",
               "--dataset",    mixed_path,
               "--preset",     f->preset_name,
               "--steps",      MICRO_TRAIN_STEPS,
               "--start-from", f->weights_path,
               "--save",       f->next_weights_path,
               "--seed",       "7",
               (char*)NULL);
        /* If exec returns, it failed */
        _exit(127);
    }
    if (pid < 0) {
        printf("  [%s] microtrain fork failed: %s\n", f->name, strerror(errno));
        return;
    }
    f->microtrain_active = 1;
    f->microtrain_pid = pid;
    printf("  [%s] microtrain corpus = %ld bytes (own holding + %d DNA samples)\n",
           f->name, mix_bytes, DNA_SAMPLES_PER_BURST);
    printf("\n  [%s] microtrain spawned (pid=%d, %ld bytes / nov %.1f / res %.1f)\n\n",
           f->name, pid, f->mass_bytes, f->mass_novelty, f->mass_resonance);
}

static void field_decay(CaveField* f) {
    f->excitement *= EXCITEMENT_DECAY;
    f->dissonance *= DISSONANCE_DECAY;
}

static int field_should_speak(const CaveField* f) {
    if (f->dissonance > TUNNEL_THRESHOLD) return 1;
    if (f->excitement > f->coherence_floor) return 1;
    return 0;
}

static void field_after_speak(CaveField* f) {
    f->spoke_count++;
    f->excitement = 0.0f;
    f->dissonance *= 0.5f;
}

static void field_maturity_drift(CaveField* f) {
    if (f->total_count < MATURITY_WINDOW) return;
    float ratio = (float)f->spoke_count / (float)f->total_count;
    if (ratio > SPEAK_RATIO_HIGH)      f->coherence_floor += MATURITY_STEP;
    else if (ratio < SPEAK_RATIO_LOW)  f->coherence_floor -= MATURITY_STEP;

    float lo = f->baseline_floor - MATURITY_CAP;
    float hi = f->baseline_floor + MATURITY_CAP;
    if (f->coherence_floor < lo) f->coherence_floor = lo;
    if (f->coherence_floor > hi) f->coherence_floor = hi;
}

/*
 * Silent generation: same core loop as generate() but returns the tokens
 * through an out-buffer instead of printing them as "you:/cave:". The caller
 * decides how to label the output ([A]/[B]/[user]).
 */
static int dual_generate(CaveModel* m, CaveVocab* vocab,
                         const int* prompt, int prompt_len,
                         int max_gen, float temp, float top_p,
                         int* out, int out_cap) {
    int Vp = vocab->vocab_size + 2;
    int ctx[MAX_SEQ];
    int ctx_len = 0, gen_len = 0;

    kv_reset();

    ctx[ctx_len++] = vocab->bos_id;
    for (int i = 0; i < prompt_len && ctx_len < CTX; i++)
        ctx[ctx_len++] = prompt[i];

    for (int i = 0; i < ctx_len - 1; i++) {
        float* logits = model_forward(m, ctx[i], i);
        free(logits);
    }

    int last_id = ctx[ctx_len - 1];
    for (int step = 0; step < max_gen && ctx_len < CTX && gen_len < out_cap; step++) {
        float* logits = model_forward(m, last_id, ctx_len - 1);

        /* Never emit BOS as the very first generated token — the cave must
         * actually open its mouth once it decided to speak. */
        if (gen_len == 0) logits[vocab->bos_id] = -1e9f;

        int next = sample_top_p(logits, Vp, temp, top_p);
        free(logits);

        if (next == vocab->bos_id) break;

        ctx[ctx_len++] = next;
        out[gen_len++] = next;
        last_id = next;
    }

    /* Post-generation Hebbian + co-occ, same as the interactive path. */
    hebbian_update(m, NULL, vocab->vocab_size + 2, out, gen_len, 0);
    count_emerged_usage(m, vocab, out, gen_len);

    int all[MAX_SEQ]; int al = 0;
    for (int i = 0; i < prompt_len && al < MAX_SEQ; i++) all[al++] = prompt[i];
    for (int i = 0; i < gen_len && al < MAX_SEQ; i++)    all[al++] = out[i];
    cooccur_update(&m->cooccur, all, al);
    cooccur_decay(&m->cooccur, 0.999f);
    check_symbol_survival(m, vocab);
    try_emerge_symbol(m, vocab);

    return gen_len;
}

static void print_glyphs(const char* label, CaveVocab* v, const int* toks, int len) {
    printf("[%s] ", label);
    for (int i = 0; i < len; i++) {
        if (toks[i] >= 0 && toks[i] < v->vocab_size)
            printf("%s ", v->tokens[toks[i]]);
    }
    printf("\n");
    fflush(stdout);
}

/* Non-blocking line read. Returns bytes read (or 0 if nothing). */
static int try_read_user_line(char* buf, int cap) {
    int n = (int)read(fileno(stdin), buf, cap - 1);
    if (n <= 0) return 0;
    buf[n] = '\0';
    char* nl = strchr(buf, '\n');
    if (nl) *nl = '\0';
    return n;
}

/*
 * Colony loop: N caves breathe on each other. Any cave that trips its
 * silence gate speaks; every other cave hears. User can drop glyphs into
 * the ring at any moment — they broadcast to every active cave. Each
 * utterance (by any cave) occasionally lands a copy in the shared dna/
 * pool so future scans of the learner can feed the ecology back into
 * every model's passive-reading path. Mitosis and pressure-death hook in
 * later — this tick loop is N-symmetric already.
 */
static void colony_main(float temp, float top_p, const char* preset_name) {
    /* preset_name is needed by colony_try_mitosis so children share the
     * founders' architecture (same-size mitosis for now; downsize later). */

    /* Non-blocking stdin — user glyphs any time. */
    int flags = fcntl(fileno(stdin), F_GETFL, 0);
    fcntl(fileno(stdin), F_SETFL, flags | O_NONBLOCK);

    int last_tokens[MAX_SEQ];
    int last_len = 0;

    /* Bootstrap: speak-from-yesterday if any cave has a persisted last_tokens
     * (spore hydrated in cave_new). Pick the cave with highest excitement
     * after wake-impulse — its yesterday becomes the ring's opening context.
     * Every other cave "hears" that awakening through field_hear (wake echo).
     * Fallback for a fully cold-start ring: the neutral BE pulse. */
    int best_ci = -1;
    float best_exc = -1.0f;
    for (int ci = 0; ci < g_colony_n; ci++) {
        if (g_colony[ci]->last_len > 0 && g_colony[ci]->field.excitement > best_exc) {
            best_exc = g_colony[ci]->field.excitement;
            best_ci = ci;
        }
    }

    if (best_ci >= 0 && best_exc > 0.0f) {
        Cave* waker = g_colony[best_ci];
        int n = waker->last_len;
        if (n > MAX_SEQ) n = MAX_SEQ;
        memcpy(last_tokens, waker->last_tokens, n * sizeof(int));
        last_len = n;
        /* Wake-echo: every cave in the ring hears the waker's yesterday
         * as passive input — wakes excitement without a forward pass. */
        for (int i = 0; i < n; i++) {
            int prev = (i == 0) ? -1 : last_tokens[i-1];
            for (int ci = 0; ci < g_colony_n; ci++)
                field_hear(&g_colony[ci]->field, g_colony[ci]->model, prev, last_tokens[i]);
        }
        printf("  [wake] %s speaks from yesterday (%d tokens, exc %.2f)\n",
               waker->field.name, n, best_exc);
    } else {
        /* Cold start — neutral BE pulse */
        int be_id = semtok_find_glyph("BE");
        if (be_id >= 0) {
            for (int ci = 0; ci < g_colony_n; ci++)
                field_hear(&g_colony[ci]->field, g_colony[ci]->model, -1, be_id);
            last_tokens[0] = be_id;
            last_len = 1;
        }
    }

    printf("══════════════════════════════════════════════════════════\n");
    printf("  caveLLMan — COLONY (n=%d)\n", g_colony_n);
    printf("══════════════════════════════════════════════════════════\n");
    for (int ci = 0; ci < g_colony_n; ci++) {
        CaveField* f = &g_colony[ci]->field;
        printf("  %s: baseline floor %.2f\n", f->name, f->baseline_floor);
    }
    printf("  tunnel=%.2f, decay=%.2f/%.2f, maturity drift ±%.2f\n",
           TUNNEL_THRESHOLD, EXCITEMENT_DECAY, DISSONANCE_DECAY, MATURITY_CAP);
    printf("  drop glyphs into the ring any time. 'quit' to exit.\n");
    start_learner("feed", DNA_DIR);
    printf("──────────────────────────────────────────────────────────\n\n");

    int dna_counter = 0;
    int running = 1;
    while (running) {
        /* Hold the learner's lock while we mutate model/field state —
         * otherwise the feed/ and dna/ scans race us on cooccur/emergence. */
        pthread_mutex_lock(&g_learner.lock);

        /* 1. User input (non-blocking) */
        char ubuf[512];
        if (try_read_user_line(ubuf, sizeof(ubuf)) > 0) {
            if (strcmp(ubuf, "quit") == 0 || strcmp(ubuf, "exit") == 0) {
                pthread_mutex_unlock(&g_learner.lock);
                break;
            }
            if (strcmp(ubuf, "stats") == 0) {
                for (int ci = 0; ci < g_colony_n; ci++) {
                    CaveField* f = &g_colony[ci]->field;
                    printf("  [%s] exc=%.2f floor=%.2f diss=%.2f spoke=%d/%d mass=%ldB/%.1fn/%.1fr\n",
                           f->name, f->excitement, f->coherence_floor, f->dissonance,
                           f->spoke_count, f->total_count,
                           f->mass_bytes, f->mass_novelty, f->mass_resonance);
                }
                pthread_mutex_unlock(&g_learner.lock);
                continue;
            }
            /* Tokenize against the first cave's vocab (all share the 88 canonical
             * glyphs, so the token ids are interchangeable for base symbols). */
            int utoks[MAX_SEQ];
            int un = tokenize_prompt(ubuf, g_colony[0]->vocab, utoks, MAX_SEQ);
            if (un > 0) {
                print_glyphs("user", g_colony[0]->vocab, utoks, un);
                int prev = (last_len > 0) ? last_tokens[last_len - 1] : -1;
                for (int i = 0; i < un; i++) {
                    int p = (i == 0) ? prev : utoks[i - 1];
                    for (int ci = 0; ci < g_colony_n; ci++)
                        field_hear(&g_colony[ci]->field, g_colony[ci]->model, p, utoks[i]);
                }
                for (int ci = 0; ci < g_colony_n; ci++)
                    field_append_holding(&g_colony[ci]->field, g_colony[ci]->vocab, utoks, un);
                memcpy(last_tokens, utoks, (size_t)un * sizeof(int));
                last_len = un;
            }
        }

        /* 2. Randomized turn order through the whole colony (Fisher-Yates) */
        int order[COLONY_MAX];
        for (int i = 0; i < g_colony_n; i++) order[i] = i;
        for (int i = g_colony_n - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }

        for (int k = 0; k < g_colony_n; k++) {
            Cave* c = g_colony[order[k]];
            CaveField* f = &c->field;

            f->total_count++;
            if (!field_should_speak(f)) continue;

            int gen[MAX_SEQ];
            int prompt_len = (last_len > 0) ? last_len : 0;
            int gn = dual_generate(c->model, c->vocab, last_tokens, prompt_len,
                                   DUAL_MAX_GEN, temp, top_p,
                                   gen, MAX_SEQ);
            if (gn <= 0) continue; /* wanted to speak but produced nothing */

            print_glyphs(f->name, c->vocab, gen, gn);
            field_after_speak(f);

            /* Every other cave in the colony hears this utterance */
            int prev = (last_len > 0) ? last_tokens[last_len - 1] : -1;
            for (int ci = 0; ci < g_colony_n; ci++) {
                if (ci == order[k]) continue; /* speaker doesn't re-hear itself */
                Cave* other = g_colony[ci];
                for (int i = 0; i < gn; i++) {
                    int p = (i == 0) ? prev : gen[i - 1];
                    field_hear(&other->field, other->model, p, gen[i]);
                }
                field_append_holding(&other->field, c->vocab, gen, gn);
            }

            /* Every 2nd utterance deposits in the shared dna/ pool so
             * the colony starts feeding on its own speech early — original
             * threshold of 5 meant a quiet autonomous ring never hit the
             * first DNA write (observed 2026-04-22: 3 utterances/hour with
             * threshold 5 → zero DNA files, zero CPT trigger, zero mitosis). */
            if ((++dna_counter % 2) == 0) dna_write_cave(c, gen, gn);

            memcpy(last_tokens, gen, (size_t)gn * sizeof(int));
            last_len = gn;
        }

        /* 3/4/5. Decay, maturity drift, microtrain, immunity countdown. */
        for (int ci = 0; ci < g_colony_n; ci++) {
            field_decay(&g_colony[ci]->field);
            field_maturity_drift(&g_colony[ci]->field);
            field_microtrain_tick(&g_colony[ci]->field, g_colony[ci]->model);
            if (g_colony[ci]->field.immunity_ticks > 0)
                g_colony[ci]->field.immunity_ticks--;
        }

        /* 5a. Spontaneous pulse — every ~500 ticks the quietest cave gets a
         * kick so autonomous rings don't collapse into silent equilibrium.
         * The kick is ADAPTIVE to the cave's baseline floor: it always
         * lifts excitement just above the cave's natural threshold
         * (baseline_floor + 0.05), so a B cave with floor 0.60 actually
         * crosses its gate, not just an A cave with floor 0.30. The first
         * 2026-04-25 Railway deploy showed the bug: hardcoded magnitude
         * 0.15 was below B's baseline 0.30 → pulse hit B repeatedly,
         * B never spoke, A excluded → entire ring went silent after 20s.
         * Preserves autonomy: no external input source, just an adaptive
         * inner heartbeat. */
        static int g_pulse_counter = 0;
        if ((++g_pulse_counter % 500) == 0 && g_colony_n > 0) {
            int quietest = 0;
            float min_ratio = 1e9f;
            for (int ci = 0; ci < g_colony_n; ci++) {
                const CaveField* f = &g_colony[ci]->field;
                float r = (f->total_count > 0)
                    ? (float)f->spoke_count / (float)f->total_count : 0.0f;
                if (r < min_ratio) { min_ratio = r; quietest = ci; }
            }
            CaveField* qf = &g_colony[quietest]->field;
            float kick = qf->baseline_floor + 0.05f;
            qf->excitement = (kick > qf->excitement) ? kick : qf->excitement;
            if (qf->excitement > EXCITEMENT_CAP) qf->excitement = EXCITEMENT_CAP;
            printf("  [pulse] %s exc->%.2f (baseline+0.05)\n", qf->name, qf->excitement);
        }

        /* 5b. Sexual mitosis — two fittest caves may spawn a child. */
        colony_try_mitosis(preset_name);

        /* 5c. Periodic spore autosave every 200 ticks (safety against crash
         * and ordinary wear — cave's current state becomes tomorrow's wake). */
        static int g_save_counter = 0;
        if ((++g_save_counter % 200) == 0) {
            for (int ci = 0; ci < g_colony_n; ci++) {
                Cave* c = g_colony[ci];
                save_cave_spore(c, last_tokens, last_len, cave_fingerprint_of(c));
            }
        }

        pthread_mutex_unlock(&g_learner.lock);

        /* Force flush stdout each tick — Railway's logging driver buffers
         * tiny writes regardless of setvbuf(_IONBF) on the C side, so
         * silence in the log stream looked like a dead ring on the first
         * deploy. Cheap call (10 Hz at our tick rate). */
        fflush(stdout);

        /* 6. Pace (outside the lock so the feed/ thread can work) */
        usleep(DUAL_TICK_US);
    }

    stop_learner();

    /* Save each cave's state before exit — this becomes tomorrow's wake. */
    printf("\n");
    for (int ci = 0; ci < g_colony_n; ci++) {
        Cave* c = g_colony[ci];
        if (save_cave_spore(c, last_tokens, last_len, cave_fingerprint_of(c)) == 0)
            printf("  [%s] spore saved.\n", c->field.name);
        else
            printf("  [%s] spore save failed.\n", c->field.name);
    }

    /* Reap any microtrain children still running at exit */
    for (int ci = 0; ci < g_colony_n; ci++) {
        CaveField* f = &g_colony[ci]->field;
        if (f->microtrain_active) waitpid(f->microtrain_pid, NULL, 0);
    }

    printf("\n");
    for (int ci = 0; ci < g_colony_n; ci++) {
        CaveField* f = &g_colony[ci]->field;
        printf("  [%s] spoke %d / %d turns, final floor %.3f, microtrains=%d\n",
               f->name, f->spoke_count, f->total_count,
               f->coherence_floor, f->microtrain_done_count);
    }
    printf("the colony remembers.\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Async ring (--async flag) — each cave runs its own pthread, ticking at
 * its own pace. Forward passes execute in parallel (TLS KV cache), shared
 * state mutations serialize through g_learner.lock. molequla shows that
 * concurrent organism processes give ~N× CPU usage on multi-core hosts;
 * caveLLMan's sync ring used at most one core because the tick loop
 * iterated caves sequentially. Async fixes that.
 *
 * Klaus-style metarecursion: after a cave speaks, it re-hears its own
 * utterance through field_hear at 15% weight. Body re-ingests own exhale
 * (klaus.c "META-RECURSION (re-inhale own output, blend 85/15)").
 * ═══════════════════════════════════════════════════════════════════════ */

static volatile int g_async_running = 1;
static int g_async_last_tokens[MAX_SEQ];
static int g_async_last_len = 0;
static int g_async_dna_counter = 0;
static pthread_t g_cave_threads[COLONY_MAX];
static int       g_cave_thread_started[COLONY_MAX];

/* Tweakable async knobs — two parallel async deploys can differ on these
 * (Railway customStartCommand passes different --metarecursion / --pulse-margin
 * values) for A/B observation. */
static float g_metarecursion = 0.15f;  /* klaus 85/15 default */
static float g_pulse_margin  = 0.05f;  /* default kick = baseline_floor + 0.05 */
static int   g_trinity_mode  = 0;      /* --trinity flag: Molly as horizon-goroutine + affair mitosis */

/* MollyState + g_molly are declared earlier (before trinity-mitosis
 * helpers that reference them). See above. */

typedef struct {
    Cave* cave;
    float temp;
    float top_p;
} CaveThreadArg;

static CaveThreadArg g_cave_thread_args[COLONY_MAX];

/* Molly's thread — she lives on the horizon, not in the colony. Her tick
 * rate is independent of cave tick. Each cycle she picks a short prompt
 * from her own last_tokens (or a random base glyph), generates a fragment,
 * writes it to dna/output/molly_<ts>.txt so the existing learner ingests
 * it and feeds every cave passively. She never takes the g_learner.lock
 * for inference — her own pthread mutex on her model state is enough.
 * Caves never see her in g_colony[]; she influences them only through
 * the field. */
#define MOLLY_TICK_US (DUAL_TICK_US * 3)   /* slower than cave threads — she breathes deeper */
#define MOLLY_DNA_DIR "dna/output/molly"
/* g_molly_lock declared earlier with MollyState. */

static void* molly_thread_main(void* arg) {
    (void)arg;
    /* Initial seed for prompt — a base BE glyph, like the bootstrap. */
    int be_id = semtok_find_glyph("BE");
    if (g_molly.last_len == 0 && be_id >= 0) {
        g_molly.last_tokens[0] = be_id;
        g_molly.last_len = 1;
    }

    mkdir("dna", 0755);
    mkdir("dna/output", 0755);
    mkdir(MOLLY_DNA_DIR, 0755);

    while (g_async_running) {
        int prompt[MAX_SEQ];
        int prompt_len;

        pthread_mutex_lock(&g_molly_lock);
        prompt_len = g_molly.last_len;
        if (prompt_len > MAX_SEQ) prompt_len = MAX_SEQ;
        memcpy(prompt, g_molly.last_tokens, (size_t)prompt_len * sizeof(int));
        pthread_mutex_unlock(&g_molly_lock);

        /* Forward pass on Molly's own model. Hold g_learner.lock — same lock
         * the cave threads hold during their dual_generate — because the KV
         * cache (kv_keys / kv_vals / kv_len) is GLOBAL static. Without this,
         * Molly's forward and a cave's forward race on the shared KV arrays,
         * which on Linux is a guaranteed SIGSEGV after a few seconds (the
         * Trinity 42-65s crash on Railway). g_molly_lock alone protected
         * Molly's *own* state but did nothing about the global KV. */
        pthread_mutex_lock(&g_learner.lock);
        pthread_mutex_lock(&g_molly_lock);
        int gen[MAX_SEQ];
        int gn = dual_generate(g_molly.model, g_molly.vocab,
                               prompt, prompt_len,
                               DUAL_MAX_GEN, 0.8f, 0.9f, gen, MAX_SEQ);
        pthread_mutex_unlock(&g_molly_lock);
        pthread_mutex_unlock(&g_learner.lock);

        if (gn > 0) {
            /* Print her voice to the log so the dashboard sees her too. */
            print_glyphs("M", g_molly.vocab, gen, gn);

            /* Write to dna/output/molly/ so the learner picks her up
             * naturally and feeds every cave through the existing
             * passive-reading path — no changes to cave_thread_main needed. */
            char fname[1024];
            snprintf(fname, sizeof(fname), "%s/molly_%ld_%d.txt",
                     MOLLY_DNA_DIR, (long)time(NULL), rand());
            FILE* f = fopen(fname, "w");
            if (f) {
                for (int i = 0; i < gn; i++) {
                    int t = gen[i];
                    if (t < 0 || t >= g_molly.vocab->vocab_size) continue;
                    fprintf(f, "%s ", g_molly.vocab->tokens[t]);
                }
                fprintf(f, ".\n");
                fclose(f);
            }

            /* Update her own last_tokens for next cycle's prompt. */
            pthread_mutex_lock(&g_molly_lock);
            int n = (gn > 32) ? 32 : gn;
            memcpy(g_molly.last_tokens, gen, (size_t)n * sizeof(int));
            g_molly.last_len = n;
            g_molly.utter_count++;
            pthread_mutex_unlock(&g_molly_lock);
        }

        usleep(MOLLY_TICK_US);
        fflush(stdout);
    }
    return NULL;
}

static void* cave_thread_main(void* arg) {
    CaveThreadArg* a = (CaveThreadArg*)arg;
    Cave* c = a->cave;
    float temp = a->temp;
    float top_p = a->top_p;

    while (g_async_running) {
        c->field.total_count++;

        if (field_should_speak(&c->field)) {
            /* Hold g_learner.lock through the entire generate-and-mutate
             * block: dual_generate's post-pass (hebbian_update, cooccur_update,
             * check_symbol_survival, try_emerge_symbol) writes the cave's
             * own emerged[], vocab, and Hebbian arrays — and the learner
             * thread modifies those same fields when it ingests feed/ or
             * dna/. Race lands on Linux NPTL and corrupts vocab_size /
             * n_emerged → SIGSEGV. Trinity hit this on Railway because M's
             * pre-loaded excitement makes her speak instantly, all three
             * caves write DNA right away, and the learner picks them up
             * within a tick.
             *
             * Trade-off: forward passes serialize again (no longer truly
             * parallel matvec across cores). Correctness > speed for now.
             * A cleaner fix splits dual_generate into pre/post halves so
             * only the post-half holds the lock — TODO. */
            int prompt[MAX_SEQ];
            int prompt_len;
            int gen[MAX_SEQ];
            int gn;

            pthread_mutex_lock(&g_learner.lock);

            prompt_len = g_async_last_len;
            if (prompt_len > MAX_SEQ) prompt_len = MAX_SEQ;
            memcpy(prompt, g_async_last_tokens, (size_t)prompt_len * sizeof(int));

            gn = dual_generate(c->model, c->vocab, prompt, prompt_len,
                               DUAL_MAX_GEN, temp, top_p, gen, MAX_SEQ);

            if (gn > 0) {
                print_glyphs(c->field.name, c->vocab, gen, gn);
                field_after_speak(&c->field);

                /* Other caves hear this utterance. */
                int prev = (g_async_last_len > 0)
                    ? g_async_last_tokens[g_async_last_len - 1]
                    : -1;
                for (int ci = 0; ci < g_colony_n; ci++) {
                    if (g_colony[ci] == c) continue;
                    Cave* other = g_colony[ci];
                    for (int i = 0; i < gn; i++) {
                        int p = (i == 0) ? prev : gen[i - 1];
                        field_hear(&other->field, other->model, p, gen[i]);
                    }
                    field_append_holding(&other->field, c->vocab, gen, gn);
                }

                /* Klaus metarecursion — cave re-hears its OWN utterance at
                 * g_metarecursion weight (default 0.15, klaus 85/15 blend). */
                for (int i = 0; i < gn; i++) {
                    int p = (i == 0) ? prev : gen[i - 1];
                    float before = c->field.excitement;
                    field_hear(&c->field, c->model, p, gen[i]);
                    float delta = c->field.excitement - before;
                    c->field.excitement = before + delta * g_metarecursion;
                }

                if ((++g_async_dna_counter % 2) == 0) dna_write_cave(c, gen, gn);

                memcpy(g_async_last_tokens, gen, (size_t)gn * sizeof(int));
                g_async_last_len = gn;
            }

            pthread_mutex_unlock(&g_learner.lock);
        }

        /* Per-cave bookkeeping — own field, no shared state, no lock. */
        field_decay(&c->field);
        field_maturity_drift(&c->field);
        field_microtrain_tick(&c->field, c->model);
        if (c->field.immunity_ticks > 0) c->field.immunity_ticks--;

        usleep(DUAL_TICK_US);
        fflush(stdout);
    }

    return NULL;
}

static void* orchestrator_thread_main(void* arg) {
    const char* preset_name = (const char*)arg;
    int pulse_counter = 0;
    int save_counter = 0;

    while (g_async_running) {
        sleep(1);
        if (!g_async_running) break;

        pthread_mutex_lock(&g_learner.lock);

        /* Pulse every 5 seconds (wall clock — independent of per-cave tick rate). */
        pulse_counter++;
        if ((pulse_counter % 5) == 0 && g_colony_n > 0) {
            int quietest = 0;
            float min_ratio = 1e9f;
            for (int ci = 0; ci < g_colony_n; ci++) {
                const CaveField* f = &g_colony[ci]->field;
                float r = (f->total_count > 0)
                    ? (float)f->spoke_count / (float)f->total_count : 0.0f;
                if (r < min_ratio) { min_ratio = r; quietest = ci; }
            }
            CaveField* qf = &g_colony[quietest]->field;
            float kick = qf->baseline_floor + g_pulse_margin;
            qf->excitement = (kick > qf->excitement) ? kick : qf->excitement;
            if (qf->excitement > EXCITEMENT_CAP) qf->excitement = EXCITEMENT_CAP;
            printf("  [pulse-async] %s exc->%.2f (baseline+%.2f)\n",
                   qf->name, qf->excitement, g_pulse_margin);
        }

        int colony_n_before = g_colony_n;
        if (g_trinity_mode) colony_try_mitosis_trinity(preset_name);
        else                colony_try_mitosis(preset_name);
        for (int ci = colony_n_before; ci < g_colony_n; ci++) {
            if (g_cave_thread_started[ci]) continue;
            g_cave_thread_args[ci].cave = g_colony[ci];
            g_cave_thread_args[ci].temp = g_cave_thread_args[0].temp;
            g_cave_thread_args[ci].top_p = g_cave_thread_args[0].top_p;
            pthread_create(&g_cave_threads[ci], NULL,
                           cave_thread_main, &g_cave_thread_args[ci]);
            g_cave_thread_started[ci] = 1;
            printf("  [async] thread spawned for %s\n", g_colony[ci]->field.name);
        }

        /* Save every 20 seconds. */
        save_counter++;
        if ((save_counter % 20) == 0) {
            for (int ci = 0; ci < g_colony_n; ci++) {
                Cave* c = g_colony[ci];
                save_cave_spore(c, g_async_last_tokens, g_async_last_len,
                               cave_fingerprint_of(c));
            }
        }

        pthread_mutex_unlock(&g_learner.lock);
        fflush(stdout);
    }
    return NULL;
}

static void colony_main_async(float temp, float top_p, const char* preset_name) {
    /* Bootstrap (same logic as sync colony_main — wake-from-yesterday OR
     * neutral BE pulse fallback). Only difference is we copy the bootstrap
     * tokens into the global g_async_last_tokens for cave threads to read. */
    int last_tokens[MAX_SEQ];
    int last_len = 0;

    int best_ci = -1;
    float best_exc = -1.0f;
    for (int ci = 0; ci < g_colony_n; ci++) {
        if (g_colony[ci]->last_len > 0 && g_colony[ci]->field.excitement > best_exc) {
            best_exc = g_colony[ci]->field.excitement;
            best_ci = ci;
        }
    }

    if (best_ci >= 0 && best_exc > 0.0f) {
        Cave* waker = g_colony[best_ci];
        int n = waker->last_len;
        if (n > MAX_SEQ) n = MAX_SEQ;
        memcpy(last_tokens, waker->last_tokens, (size_t)n * sizeof(int));
        last_len = n;
        for (int i = 0; i < n; i++) {
            int prev = (i == 0) ? -1 : last_tokens[i-1];
            for (int ci = 0; ci < g_colony_n; ci++)
                field_hear(&g_colony[ci]->field, g_colony[ci]->model, prev, last_tokens[i]);
        }
        printf("  [wake] %s speaks from yesterday (%d tokens, exc %.2f)\n",
               waker->field.name, n, best_exc);
    } else {
        int be_id = semtok_find_glyph("BE");
        if (be_id >= 0) {
            for (int ci = 0; ci < g_colony_n; ci++)
                field_hear(&g_colony[ci]->field, g_colony[ci]->model, -1, be_id);
            last_tokens[0] = be_id;
            last_len = 1;
        }
    }

    memcpy(g_async_last_tokens, last_tokens, (size_t)last_len * sizeof(int));
    g_async_last_len = last_len;
    g_async_dna_counter = 0;
    g_async_running = 1;

    printf("══════════════════════════════════════════════════════════\n");
    printf("  caveLLMan — ASYNC RING (n=%d, threads-per-cave)\n", g_colony_n);
    printf("══════════════════════════════════════════════════════════\n");
    for (int ci = 0; ci < g_colony_n; ci++) {
        CaveField* f = &g_colony[ci]->field;
        printf("  %s: baseline floor %.2f\n", f->name, f->baseline_floor);
    }
    printf("  tunnel=%.2f, decay=%.2f/%.2f, maturity drift +-%.2f\n",
           TUNNEL_THRESHOLD, EXCITEMENT_DECAY, DISSONANCE_DECAY, MATURITY_CAP);
    printf("  klaus metarecursion: cave re-hears own exhale @ 15%% weight\n");
    start_learner("feed", DNA_DIR);
    printf("──────────────────────────────────────────────────────────\n\n");

    /* Spawn one pthread per cave. */
    for (int ci = 0; ci < g_colony_n; ci++) {
        g_cave_thread_args[ci].cave = g_colony[ci];
        g_cave_thread_args[ci].temp = temp;
        g_cave_thread_args[ci].top_p = top_p;
        pthread_create(&g_cave_threads[ci], NULL,
                       cave_thread_main, &g_cave_thread_args[ci]);
        g_cave_thread_started[ci] = 1;
    }

    /* Trinity: Molly's horizon-thread (her own rhythm, writes to dna/ pool).
     * DIAG: temporarily disabled to confirm whether the Linux Trinity 30-65s
     * silent crash is in Molly's thread or in the cave/orchestrator path. */
    if (g_trinity_mode && g_molly.model) {
        // pthread_create(&g_molly.thread, NULL, molly_thread_main, NULL);
        // g_molly.started = 1;
        printf("[trinity] Molly thread DIAG-DISABLED (model loaded but not running)\n");
    }

    /* Orchestrator: pulse / mitosis / save / spawn-thread-for-new-cave. */
    pthread_t orch;
    pthread_create(&orch, NULL, orchestrator_thread_main, (void*)preset_name);

    /* Main thread polls stdin for quit signal. */
    int flags = fcntl(fileno(stdin), F_GETFL, 0);
    fcntl(fileno(stdin), F_SETFL, flags | O_NONBLOCK);

    char ubuf[512];
    while (g_async_running) {
        if (try_read_user_line(ubuf, sizeof(ubuf)) > 0) {
            if (strcmp(ubuf, "quit") == 0 || strcmp(ubuf, "exit") == 0) {
                g_async_running = 0;
                break;
            }
        }
        sleep(1);
    }

    g_async_running = 0;

    /* Join all threads. */
    for (int ci = 0; ci < g_colony_n; ci++) {
        if (g_cave_thread_started[ci]) {
            pthread_join(g_cave_threads[ci], NULL);
        }
    }
    if (g_molly.started) pthread_join(g_molly.thread, NULL);
    pthread_join(orch, NULL);

    stop_learner();

    printf("\n");
    for (int ci = 0; ci < g_colony_n; ci++) {
        Cave* c = g_colony[ci];
        if (save_cave_spore(c, g_async_last_tokens, g_async_last_len,
                            cave_fingerprint_of(c)) == 0)
            printf("  [%s] spore saved.\n", c->field.name);
    }

    for (int ci = 0; ci < g_colony_n; ci++) {
        CaveField* f = &g_colony[ci]->field;
        if (f->microtrain_active) waitpid(f->microtrain_pid, NULL, 0);
    }

    printf("\n");
    for (int ci = 0; ci < g_colony_n; ci++) {
        CaveField* f = &g_colony[ci]->field;
        printf("  [%s] spoke %d / %d turns, final floor %.3f, microtrains=%d\n",
               f->name, f->spoke_count, f->total_count,
               f->coherence_floor, f->microtrain_done_count);
    }
    printf("the colony remembers (async).\n");
}

/* ── Persistence layer — per-cave binary spore ──────────────────────────
 *
 * Brings back the pre-colony save/load layer that was dropped during the
 * ring pivot, extended with colony-aware semantics. Per-cave binary file
 * weights/cavellman_<name>.spore. Fingerprint gates loading against the
 * wrong weights (DoE mycelium convention). CRC32 tail gates against
 * corrupted files. No SQLite — ring N≤8, binary spore is the right grain.
 *
 * What we persist: cooccur matrix, emerged symbols (+ their names),
 * Hebbian LoRA adapters, CaveField, last_tokens, saved_at.
 * What we don't: excitement/dissonance (transient, rebuilt by wake),
 * KV cache (transient), holding buffers (already on disk as files).
 *
 * Functions are defined here but NOT YET wired into cave_new /
 * colony_main / colony_mitosis. Wire-in happens after the current
 * observation dry-run completes. See project_cavellman_persistence.md.
 */

#define CAVE_SPORE_MAGIC    0x45564143u     /* 'CAVE' */
#define CAVE_SPORE_VERSION  1u
#define CAVE_LAST_TOKENS    32              /* "what was on mind when sleeping" */

typedef struct {
    uint32_t magic;
    uint32_t version;
    int64_t  saved_at;
    uint64_t weights_fingerprint;
    int32_t  n_layers;
    int32_t  E;
    int32_t  hebbian_rank;
    int32_t  base_vocab;
    int32_t  n_emerged;
    int32_t  last_len;
    /* CaveField POD mirror */
    float    baseline_floor;
    float    coherence_floor;
    int32_t  spoke_count;
    int32_t  total_count;
    int64_t  mass_bytes;
    float    mass_novelty;
    float    mass_resonance;
    int32_t  microtrain_done_count;
    int32_t  immunity_ticks;
    uint32_t _pad;  /* keep 8-byte alignment for what follows */
} CaveSporeHeader;

/* CRC32 (IEEE) — one-shot, small table built once. */
static uint32_t cave_crc32_table[256];
static int      cave_crc32_ready = 0;

static void cave_crc32_init(void) {
    if (cave_crc32_ready) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
        cave_crc32_table[i] = c;
    }
    cave_crc32_ready = 1;
}

static uint32_t cave_crc32(const uint8_t* buf, size_t n) {
    cave_crc32_init();
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; i++)
        c = cave_crc32_table[(c ^ buf[i]) & 0xff] ^ (c >> 8);
    return c ^ 0xffffffffu;
}

/* Fingerprint weights: FNV-1a 64 over first min(n,4) tensors' len + first
 * 16 raw float words. Stable across save/load if the same .bin backs the
 * cave; changes on retrain. */
static uint64_t cave_weights_fingerprint(nt_tensor** tensors, int n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    int k = (n < 4) ? n : 4;
    for (int i = 0; i < k; i++) {
        long l = tensors[i]->len;
        h ^= (uint64_t)l; h *= 0x100000001b3ULL;
        int take = (l < 16) ? (int)l : 16;
        for (int j = 0; j < take; j++) {
            uint32_t v;
            memcpy(&v, &tensors[i]->data[j], 4);
            h ^= v; h *= 0x100000001b3ULL;
        }
    }
    return h;
}

/* L2 norm of all Hebbian adapters on the cave (plasticity density). */
static float cave_hebbian_l2_norm(const CaveModel* m) {
    double sum = 0.0;
    int rank = HEBBIAN_RANK;
    for (int l = 0; l < m->N_L; l++) {
        const Layer* ly = &m->layers[l];
        long per = (long)m->E * rank;
        const float* arrs[4] = { ly->heb_A_q, ly->heb_B_q, ly->heb_A_v, ly->heb_B_v };
        for (int a = 0; a < 4; a++) {
            if (!arrs[a]) continue;
            for (long i = 0; i < per; i++) sum += (double)arrs[a][i] * arrs[a][i];
        }
    }
    return (float)sqrt(sum);
}

/* Fraction of emerged symbols currently alive (alive==1, не frozen==2 и не dead==0). */
static float cave_emerged_alive_ratio(const CaveModel* m) {
    if (m->n_emerged <= 0) return 0.0f;
    int alive = 0;
    for (int i = 0; i < m->n_emerged; i++)
        if (m->emerged[i].alive == 1) alive++;
    return (float)alive / (float)m->n_emerged;
}

/* Build the spore path from the cave's name. Caller owns the buffer.
 * Honor CAVELLMAN_SPORE_DIR env var when set (Railway / Docker volume mount);
 * fall back to weights/ for local Mac runs. mkdir is best-effort — if the
 * directory already exists or is on a mounted volume, EEXIST is fine. */
static void cave_spore_path(const char* name, char* out, size_t out_sz) {
    const char* dir = getenv("CAVELLMAN_SPORE_DIR");
    if (dir && dir[0]) {
        mkdir(dir, 0755);
        snprintf(out, out_sz, "%s/cavellman_%s.spore", dir, name);
    } else {
        snprintf(out, out_sz, "weights/cavellman_%s.spore", name);
    }
}

/*
 * save_cave_spore — serialize cave state to disk.
 * Layout: header (fixed) | cooccur.matrix | cooccur.pair_count | total_interactions
 *       | emerged[] (POD) | emerged_names[n_emerged][32]
 *       | for each layer: heb_A_q, heb_B_q, heb_A_v, heb_B_v (each E*rank floats)
 *       | last_tokens[CAVE_LAST_TOKENS]
 *       | crc32 over everything above
 * Returns 0 on success, -1 on failure.
 */
static int save_cave_spore(const Cave* c, const int* last_tokens, int last_len,
                            uint64_t fingerprint) {
    if (!c || !c->model || !c->vocab) return -1;
    CaveModel* m = c->model;
    const CaveField* f = &c->field;

    char path[512];
    cave_spore_path(f->name, path, sizeof(path));
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);

    FILE* fp = fopen(tmp, "wb");
    if (!fp) return -1;

    /* Header */
    CaveSporeHeader h;
    memset(&h, 0, sizeof(h));
    h.magic = CAVE_SPORE_MAGIC;
    h.version = CAVE_SPORE_VERSION;
    h.saved_at = (int64_t)time(NULL);
    h.weights_fingerprint = fingerprint;
    h.n_layers = m->N_L;
    h.E = m->E;
    h.hebbian_rank = HEBBIAN_RANK;
    h.base_vocab = c->vocab->base_size;
    h.n_emerged = m->n_emerged;
    h.last_len = (last_len < 0) ? 0 : (last_len > CAVE_LAST_TOKENS ? CAVE_LAST_TOKENS : last_len);
    h.baseline_floor = f->baseline_floor;
    h.coherence_floor = f->coherence_floor;
    h.spoke_count = f->spoke_count;
    h.total_count = f->total_count;
    h.mass_bytes = f->mass_bytes;
    h.mass_novelty = f->mass_novelty;
    h.mass_resonance = f->mass_resonance;
    h.microtrain_done_count = f->microtrain_done_count;
    h.immunity_ticks = f->immunity_ticks;

    /* Write + accumulate CRC */
    uint32_t crc = 0xffffffffu;
    #define WRITE_CRC(buf, n) do { \
        fwrite((buf), 1, (n), fp); \
        const uint8_t* _b = (const uint8_t*)(buf); \
        for (size_t _i = 0; _i < (size_t)(n); _i++) \
            crc = cave_crc32_table[(crc ^ _b[_i]) & 0xff] ^ (crc >> 8); \
    } while (0)
    cave_crc32_init();

    WRITE_CRC(&h, sizeof(h));

    /* Cooccur */
    WRITE_CRC(m->cooccur.matrix, sizeof(m->cooccur.matrix));
    WRITE_CRC(m->cooccur.pair_count, sizeof(m->cooccur.pair_count));
    WRITE_CRC(&m->cooccur.total_interactions, sizeof(int));
    WRITE_CRC(&m->cooccur.last_emergence, sizeof(int));

    /* Emerged[] */
    if (m->n_emerged > 0) {
        WRITE_CRC(m->emerged, sizeof(EmergedSymbol) * m->n_emerged);
        /* emerged names live in vocab->tokens[base_size + i] — copy them too */
        for (int i = 0; i < m->n_emerged; i++) {
            char name_slot[32];
            memset(name_slot, 0, 32);
            int id = c->vocab->base_size + i;
            if (id < c->vocab->vocab_size) strncpy(name_slot, c->vocab->tokens[id], 31);
            WRITE_CRC(name_slot, 32);
        }
    }

    /* Hebbian */
    int rank = HEBBIAN_RANK;
    long per = (long)m->E * rank;
    for (int l = 0; l < m->N_L; l++) {
        const Layer* ly = &m->layers[l];
        WRITE_CRC(ly->heb_A_q, per * sizeof(float));
        WRITE_CRC(ly->heb_B_q, per * sizeof(float));
        WRITE_CRC(ly->heb_A_v, per * sizeof(float));
        WRITE_CRC(ly->heb_B_v, per * sizeof(float));
    }

    /* last_tokens — always write CAVE_LAST_TOKENS ints, zero-padded */
    int last_buf[CAVE_LAST_TOKENS];
    memset(last_buf, 0, sizeof(last_buf));
    for (int i = 0; i < h.last_len && i < CAVE_LAST_TOKENS; i++)
        last_buf[i] = last_tokens ? last_tokens[i] : 0;
    WRITE_CRC(last_buf, sizeof(last_buf));

    #undef WRITE_CRC
    crc ^= 0xffffffffu;

    /* CRC tail (raw, not part of running CRC) */
    fwrite(&crc, 4, 1, fp);
    fclose(fp);

    /* Atomic replace */
    if (rename(tmp, path) != 0) { remove(tmp); return -1; }
    return 0;
}

/*
 * load_cave_spore — read and populate the given cave in-place.
 * Checks magic / version / fingerprint / CRC. On any mismatch: silent
 * cold start (return -1, leave cave state untouched).
 * last_tokens_out / last_len_out receive the restored last_tokens.
 */
static int load_cave_spore(Cave* c, uint64_t expected_fingerprint,
                            int* last_tokens_out, int* last_len_out) {
    if (!c || !c->model || !c->vocab) return -1;
    CaveModel* m = c->model;
    CaveField* f = &c->field;

    char path[512];
    cave_spore_path(f->name, path, sizeof(path));
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize < (long)(sizeof(CaveSporeHeader) + 4)) { fclose(fp); return -1; }

    /* Slurp whole file (spore ~520KB — comfortable). */
    uint8_t* buf = (uint8_t*)malloc((size_t)fsize);
    if (!buf) { fclose(fp); return -1; }
    if (fread(buf, 1, (size_t)fsize, fp) != (size_t)fsize) { free(buf); fclose(fp); return -1; }
    fclose(fp);

    /* CRC check */
    uint32_t saved_crc;
    memcpy(&saved_crc, buf + fsize - 4, 4);
    uint32_t calc_crc = cave_crc32(buf, (size_t)fsize - 4);
    if (saved_crc != calc_crc) { free(buf); return -1; }

    size_t off = 0;
    CaveSporeHeader h;
    if (off + sizeof(h) > (size_t)fsize - 4) { free(buf); return -1; }
    memcpy(&h, buf + off, sizeof(h)); off += sizeof(h);

    if (h.magic != CAVE_SPORE_MAGIC || h.version != CAVE_SPORE_VERSION) {
        free(buf); return -1;
    }
    if (h.weights_fingerprint != expected_fingerprint) { free(buf); return -1; }
    if (h.n_layers != m->N_L || h.E != m->E || h.hebbian_rank != HEBBIAN_RANK) {
        free(buf); return -1;
    }
    if (h.base_vocab != c->vocab->base_size) { free(buf); return -1; }
    if (h.n_emerged < 0 || h.n_emerged > MAX_EMERGED) { free(buf); return -1; }

    /* Restore CaveField scalars (not excitement/dissonance — those wake-impulse fills) */
    f->baseline_floor = h.baseline_floor;
    f->coherence_floor = h.coherence_floor;
    f->spoke_count = h.spoke_count;
    f->total_count = h.total_count;
    f->mass_bytes = h.mass_bytes;
    f->mass_novelty = h.mass_novelty;
    f->mass_resonance = h.mass_resonance;
    f->microtrain_done_count = h.microtrain_done_count;
    f->immunity_ticks = h.immunity_ticks;

    /* Cooccur */
    if (off + sizeof(m->cooccur.matrix) + sizeof(m->cooccur.pair_count) + 2*sizeof(int) > (size_t)fsize - 4) {
        free(buf); return -1;
    }
    memcpy(m->cooccur.matrix, buf + off, sizeof(m->cooccur.matrix)); off += sizeof(m->cooccur.matrix);
    memcpy(m->cooccur.pair_count, buf + off, sizeof(m->cooccur.pair_count)); off += sizeof(m->cooccur.pair_count);
    memcpy(&m->cooccur.total_interactions, buf + off, sizeof(int)); off += sizeof(int);
    memcpy(&m->cooccur.last_emergence, buf + off, sizeof(int)); off += sizeof(int);

    /* Emerged[] */
    if (h.n_emerged > 0) {
        size_t need = sizeof(EmergedSymbol) * h.n_emerged + 32 * h.n_emerged;
        if (off + need > (size_t)fsize - 4) { free(buf); return -1; }
        memcpy(m->emerged, buf + off, sizeof(EmergedSymbol) * h.n_emerged);
        off += sizeof(EmergedSymbol) * h.n_emerged;
        /* Restore emerged names into vocab slots base_size..base_size+n_emerged */
        for (int i = 0; i < h.n_emerged; i++) {
            int id = c->vocab->base_size + i;
            if (id < MAX_VOCAB) {
                memcpy(c->vocab->tokens[id], buf + off, 32);
                c->vocab->tokens[id][31] = '\0';
            }
            off += 32;
        }
        c->vocab->vocab_size = c->vocab->base_size + h.n_emerged;
    }
    m->n_emerged = h.n_emerged;

    /* Hebbian */
    int rank = HEBBIAN_RANK;
    long per = (long)m->E * rank;
    size_t heb_bytes = (size_t)m->N_L * 4 * per * sizeof(float);
    if (off + heb_bytes > (size_t)fsize - 4) { free(buf); return -1; }
    for (int l = 0; l < m->N_L; l++) {
        Layer* ly = &m->layers[l];
        memcpy(ly->heb_A_q, buf + off, per * sizeof(float)); off += per * sizeof(float);
        memcpy(ly->heb_B_q, buf + off, per * sizeof(float)); off += per * sizeof(float);
        memcpy(ly->heb_A_v, buf + off, per * sizeof(float)); off += per * sizeof(float);
        memcpy(ly->heb_B_v, buf + off, per * sizeof(float)); off += per * sizeof(float);
    }

    /* last_tokens */
    int last_buf[CAVE_LAST_TOKENS];
    if (off + sizeof(last_buf) > (size_t)fsize - 4) { free(buf); return -1; }
    memcpy(last_buf, buf + off, sizeof(last_buf)); off += sizeof(last_buf);
    int ll = h.last_len;
    if (ll < 0) ll = 0;
    if (ll > CAVE_LAST_TOKENS) ll = CAVE_LAST_TOKENS;
    if (last_tokens_out) for (int i = 0; i < ll; i++) last_tokens_out[i] = last_buf[i];
    if (last_len_out) *last_len_out = ll;

    /* saved_at → cave's own slot, consumed by cave_wake_impulse */
    c->spore_saved_at = h.saved_at;

    free(buf);
    return 0;
}

/*
 * cave_wake_impulse — Oleg's idea, 2026-04-22.
 * Engine starts → metrics fire → weights nudge field → unpredictable
 * combination drops cave out of cold start with individual excitement /
 * dissonance. Cave that speaks first is the one with highest wake
 * intensity — its own yesterday decides who opens the mouth.
 *
 * Called after load_cave_spore, before first ring tick. saved_at comes
 * from the loaded header (seconds since epoch). If the cave cold-started
 * (no spore), wake_impulse is a no-op — the existing BE-pulse path takes
 * over.
 */
static float cave_wake_impulse(Cave* c, int64_t saved_at) {
    if (!c || !c->model) return 0.0f;
    CaveField* f = &c->field;
    CaveModel* m = c->model;

    if (saved_at <= 0) return 0.0f;  /* cold start — no wake needed */

    float heb_norm = cave_hebbian_l2_norm(m);
    float alive_ratio = cave_emerged_alive_ratio(m);
    float recency_h = (float)((int64_t)time(NULL) - saved_at) / 3600.0f;
    if (recency_h < 0.0f) recency_h = 0.0f;
    float fatigue = expf(-recency_h * 0.1f);

    float wake = fatigue * (0.3f + 0.4f * tanhf(heb_norm) + 0.3f * alive_ratio);
    f->excitement = wake * 0.5f;
    f->dissonance = wake * 0.2f * (1.0f - fatigue);

    /* Sleep decay cooccur — forgetting what wasn't repeated. Clamp so a
     * month-long sleep doesn't zero everything. */
    float decay = 1.0f - fminf(recency_h * 0.01f, 0.5f);
    cooccur_decay(&m->cooccur, decay);

    return wake;
}

/*
 * blend_spores — used by mitosis. Parents' spores combine into the
 * child's initial state so child doesn't boot with a blank surface
 * language. Call after blend_weights / blend_weights_downsized has
 * produced the child .bin, and after cave_new has loaded the child.
 * Parents must have current spores on disk (spore-before-mitosis save
 * is the caller's responsibility).
 */
static int blend_spores(const Cave* pa, const Cave* pb, Cave* child) {
    if (!pa || !pb || !child || !child->model || !child->vocab) return -1;
    CaveModel* ma = pa->model;
    CaveModel* mb = pb->model;
    CaveModel* mc = child->model;

    /* Cooccur → averaged */
    for (int i = 0; i < COOCCUR_SIZE; i++) {
        for (int j = 0; j < COOCCUR_SIZE; j++) {
            mc->cooccur.matrix[i][j] = 0.5f * (ma->cooccur.matrix[i][j] + mb->cooccur.matrix[i][j]);
            mc->cooccur.pair_count[i][j] = (ma->cooccur.pair_count[i][j] + mb->cooccur.pair_count[i][j]) / 2;
        }
    }
    mc->cooccur.total_interactions =
        (ma->cooccur.total_interactions + mb->cooccur.total_interactions) / 2;

    /* Emerged[] → union unique (by glyph_a/glyph_b pair), use_counts averaged */
    int nc = 0;
    EmergedSymbol combined[MAX_EMERGED];
    const EmergedSymbol* src[2] = { ma->emerged, mb->emerged };
    int n_src[2] = { ma->n_emerged, mb->n_emerged };
    for (int s = 0; s < 2; s++) {
        for (int i = 0; i < n_src[s] && nc < MAX_EMERGED; i++) {
            const EmergedSymbol* e = &src[s][i];
            /* dedup */
            int dup = -1;
            for (int k = 0; k < nc; k++) {
                if ((combined[k].glyph_a == e->glyph_a && combined[k].glyph_b == e->glyph_b) ||
                    (combined[k].glyph_a == e->glyph_b && combined[k].glyph_b == e->glyph_a)) {
                    dup = k; break;
                }
            }
            if (dup >= 0) {
                combined[dup].use_count = (combined[dup].use_count + e->use_count) / 2;
                if (e->alive > combined[dup].alive) combined[dup].alive = e->alive;
                combined[dup].strength = fmaxf(combined[dup].strength, e->strength);
            } else {
                combined[nc++] = *e;
            }
        }
    }
    if (nc > MAX_EMERGED) nc = MAX_EMERGED;
    for (int i = 0; i < nc; i++) mc->emerged[i] = combined[i];
    mc->n_emerged = nc;
    /* mirror names into vocab */
    for (int i = 0; i < nc; i++) {
        int id = child->vocab->base_size + i;
        if (id < MAX_VOCAB)
            strncpy(child->vocab->tokens[id], mc->emerged[i].name, 31);
    }
    child->vocab->vocab_size = child->vocab->base_size + nc;

    /* Hebbian → layer-wise contiguous blend (even layers from pa, odd from pb),
     * matching the weight blend convention. Child may have fewer layers than
     * parents (downsize); extra parent layers simply don't contribute. */
    int rank = HEBBIAN_RANK;
    long per = (long)mc->E * rank;
    for (int l = 0; l < mc->N_L; l++) {
        CaveModel* src_m = (l % 2 == 0) ? ma : mb;
        if (l >= src_m->N_L) continue;
        Layer* dst = &mc->layers[l];
        const Layer* sl = &src_m->layers[l];
        /* If E matches → memcpy; otherwise (downsized) prefix-truncate like
         * weight blend. HD gate in colony_mitosis already blocks downsize
         * when HD_p ≠ HD_c, so at this point src_m->E == mc->E. */
        if (src_m->E == mc->E) {
            memcpy(dst->heb_A_q, sl->heb_A_q, per * sizeof(float));
            memcpy(dst->heb_B_q, sl->heb_B_q, per * sizeof(float));
            memcpy(dst->heb_A_v, sl->heb_A_v, per * sizeof(float));
            memcpy(dst->heb_B_v, sl->heb_B_v, per * sizeof(float));
        }
    }

    /* CaveField: baseline_floor уже averaged в colony_mitosis; counts/mass
     * сброшены (newborn); immunity_ticks выставлен caller'ом. */

    return 0;
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    /* Keep stdout unbuffered even when redirected through a pipe —
     * otherwise dual-mode dialogue disappears into libc buffers until quit. */
    setvbuf(stdout, NULL, _IONBF, 0);

    /* Dual is the only mode. Single-dialogue was deprecated — the human is
     * no longer the center of the ring. Default weights are the asymmetric
     * A/B pair (Dracula extrovert + Frankenstein introvert). */
    const char* weights_a = "weights/cavellman_A.bin";
    const char* weights_b = "weights/cavellman_B.bin";
    const char* weights_m = "weights/cavellman_M.bin";   /* Molly, used in --trinity */
    const char* preset_name = "small";
    float temp = 0.8f, top_p = 0.9f;
    int seed = 42;
    int async_mode = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--weights") == 0 && i+1 < argc) {
            /* Shared weights for both engines (override A/B defaults). */
            weights_a = weights_b = argv[++i];
        }
        else if (strcmp(argv[i], "--preset") == 0 && i+1 < argc) preset_name = argv[++i];
        else if (strcmp(argv[i], "--temp") == 0 && i+1 < argc) temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i+1 < argc) top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--weights-a") == 0 && i+1 < argc) weights_a = argv[++i];
        else if (strcmp(argv[i], "--weights-b") == 0 && i+1 < argc) weights_b = argv[++i];
        else if (strcmp(argv[i], "--async") == 0) async_mode = 1;
        else if (strcmp(argv[i], "--metarecursion") == 0 && i+1 < argc) g_metarecursion = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--pulse-margin") == 0 && i+1 < argc) g_pulse_margin = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--trinity") == 0) { async_mode = 1; g_trinity_mode = 1; }
        else if (strcmp(argv[i], "--weights-m") == 0 && i+1 < argc) weights_m = argv[++i];
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("cavellman — self-evolving hieroglyphic language model (dual-only)\n\n");
            printf("  --weights FILE         Shared weights for both engines\n");
            printf("  --weights-a FILE       A's weights (extrovert, coherence_floor 0.30)\n");
            printf("  --weights-b FILE       B's weights (introvert, coherence_floor 0.60)\n");
            printf("  --preset NAME          Model preset (default: small)\n");
            printf("  --temp FLOAT           Temperature (default: 0.8)\n");
            printf("  --top-p FLOAT          Nucleus sampling (default: 0.9)\n");
            printf("  --seed N               RNG seed (default: 42)\n");
            printf("  --async                Each cave runs its own pthread (parallel forward passes)\n");
            printf("  --metarecursion FLOAT  Async only: cave's own-exhale re-hearing weight (default 0.15)\n");
            printf("  --pulse-margin FLOAT   Async only: pulse magnitude = baseline_floor + this (default 0.05)\n\n");
            printf("Default: two caves (A=Dracula extrovert, B=Frankenstein introvert) talk.\n");
            printf("You can type glyphs into the ring, or stay silent — the cave does not need you.\n");
            return 0;
        }
    }

    /* Apply preset */
    const Preset* pr = NULL;
    for (int i = 0; i < N_PRESETS; i++)
        if (strcmp(PRESETS[i].name, preset_name) == 0) { pr = &PRESETS[i]; break; }
    if (!pr) { printf("Unknown preset: %s\n", preset_name); return 1; }
    E = pr->embd; H = pr->heads; HD = E / H; FFN_D = 4 * E; N_L = pr->layers; CTX = pr->ctx;
    srand(seed);

    /* ── Seed the colony with the two founders, then run the ring ──────── */
    nt_seed(seed);
    Cave* A = cave_new("A", 0.30f, weights_a, preset_name);   /* extrovert */
    if (!A) { printf("Failed to load founder A from %s\n", weights_a); return 1; }
    Cave* B = cave_new("B", 0.60f, weights_b, preset_name);   /* introvert */
    if (!B) { printf("Failed to load founder B from %s\n", weights_b); cave_free(A); return 1; }
    A->is_founder = 1;
    B->is_founder = 1;
    colony_add(A);
    colony_add(B);

    /* Trinity mode — Molly lives on the horizon as her own goroutine.
     * She is NOT a founder, NOT in g_colony[]. Her thread continuously
     * generates utterances and writes to dna/output/molly/ so the
     * existing learner picks them up and feeds every cave passively
     * through the field — caves only ever feel her, never address her.
     * Affair mitosis uses g_molly.model directly. */
    if (g_trinity_mode) {
        /* Load Molly's vocab + model into g_molly. cave_new is the simplest
         * way to get all the per-cave allocation done; we then steal the
         * pieces we need and free the wrapper. */
        Cave* M_loader = cave_new("M", 0.20f, weights_m, preset_name);
        if (!M_loader) {
            printf("Failed to load Molly from %s — drop --trinity or train her first\n", weights_m);
            cave_free(A); cave_free(B); return 1;
        }
        g_molly.model = M_loader->model;
        g_molly.vocab = M_loader->vocab;
        strncpy(g_molly.weights_path, weights_m, sizeof(g_molly.weights_path) - 1);
        g_molly.last_len = 0;
        g_molly.utter_count = 0;
        /* M_loader's CaveField + thread args are abandoned; we only kept
         * its model + vocab pointers. M_loader itself is leaked but it's
         * a single allocation per process lifetime — acceptable. */
        free(M_loader);  /* free the shell, keep model + vocab. */

        /* Pre-load A and B with mild dissonance — they sense her presence
         * over the horizon from t=0. */
        A->field.dissonance = 0.15f;
        B->field.dissonance = 0.15f;

        printf("[trinity] Molly loaded — lives on the horizon (own pthread, dna/output/molly/)\n");
        printf("[trinity] pre-tension: A.diss=0.15, B.diss=0.15\n");
    }

    if (async_mode)
        colony_main_async(temp, top_p, preset_name);
    else
        colony_main(temp, top_p, preset_name);

    for (int ci = 0; ci < g_colony_n; ci++) cave_free(g_colony[ci]);
    g_colony_n = 0;
    return 0;
}
