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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

/* ── Limits ─────────────────────────────────────────────────────────────── */

#define MAX_VOCAB       256     /* 88 base + room for emerged symbols */
#define MAX_SEQ         128
#define MAX_EMERGED     64      /* max new symbols the model can create */
#define COOCCUR_SIZE    256     /* co-occurrence matrix dimension */
#define HEBBIAN_RANK    4       /* low-rank Hebbian LoRA (Sage 3: 8 too high for 96-dim) */
#define EMERGE_THRESHOLD 0.85f  /* co-occurrence threshold for emergence */
#define DISCIPLINE_WINDOW 100   /* interactions between emergence attempts */

/* ── Architecture (matches infer_emolm.c presets) ───────────────────────── */

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

#define SURVIVAL_USES    20   /* must be used 20 times... */
#define SURVIVAL_WINDOW  200  /* ...within 200 interactions of birth, or die */
#define MAX_DEPTH        3    /* depth cap — beyond this, freeze as new primitive */

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

    /* Hebbian state */
    CoOccurrence cooccur;
    EmergedSymbol emerged[MAX_EMERGED];
    int n_emerged;
    float hebbian_lr;
    float hebbian_decay;
} CaveModel;

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
 * If a symbol hasn't been used SURVIVAL_USES times within SURVIVAL_WINDOW
 * interactions of birth, it dies. If it survives and depth == MAX_DEPTH,
 * freeze it — it becomes a new primitive.
 */
static void check_symbol_survival(CaveModel* model, CaveVocab* vocab) {
    int now = model->cooccur.total_interactions;
    for (int i = 0; i < model->n_emerged; i++) {
        EmergedSymbol* sym = &model->emerged[i];
        if (sym->alive != 1) continue; /* skip dead or frozen */

        int age = now - sym->born_at;
        if (age >= SURVIVAL_WINDOW) {
            if (sym->use_count < SURVIVAL_USES) {
                /* Death — not used enough */
                sym->alive = 0;
                printf("  *** SYMBOL DIED: %s (used %d/%d times in %d interactions) ***\n",
                       sym->name, sym->use_count, SURVIVAL_USES, age);
            } else if (sym->depth >= MAX_DEPTH) {
                /* Freeze — survived at max depth, becomes primitive */
                sym->alive = 2;
                printf("  *** SYMBOL FROZEN: %s → new primitive (depth %d, used %d times) ***\n",
                       sym->name, sym->depth, sym->use_count);
            }
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

static CaveModel* model_load(const char* weights_path, int V) {
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

    printf("  loaded %d/%d tensors, Hebbian rank=%d\n", pi, n_loaded, HEBBIAN_RANK);
    free(loaded);
    return m;
}

/* ── Forward pass (single token, with KV cache + Hebbian) ──────────────── */

/* KV cache */
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
    for (int i = 0; i < nout; i++) {
        float s = 0;
        for (int j = 0; j < nin; j++) s += W[i * nin + j] * x[j];
        out[i] = s;
    }
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
    float x[256], xn[256], q[256], k[256], v[256];
    float attn_out[256], fc1[1024], fc2[256], proj[256];

    /* Embed */
    for (int j = 0; j < E; j++)
        x[j] = m->wte->data[token_id * E + j] + m->wpe->data[pos * E + j];

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

        /* Store KV */
        memcpy(kv_keys[l][pos], k, E * sizeof(float));
        memcpy(kv_vals[l][pos], v, E * sizeof(float));

        /* Multi-head attention */
        memset(attn_out, 0, E * sizeof(float));
        float scale = 1.0f / sqrtf((float)HD);
        for (int h = 0; h < H; h++) {
            int hs = h * HD;
            float scores[128];
            for (int t = 0; t <= pos; t++) {
                float dot = 0;
                for (int j = 0; j < HD; j++) dot += q[hs+j] * kv_keys[l][t][hs+j];
                scores[t] = dot * scale;
            }
            /* Softmax */
            float mx = scores[0];
            for (int t = 1; t <= pos; t++) if (scores[t] > mx) mx = scores[t];
            float sm = 0;
            for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t] - mx); sm += scores[t]; }
            for (int t = 0; t <= pos; t++) scores[t] /= sm;
            /* Weighted sum */
            for (int j = 0; j < HD; j++) {
                float val = 0;
                for (int t = 0; t <= pos; t++) val += scores[t] * kv_vals[l][t][hs+j];
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
    int Vp = MAX_VOCAB;
    float* logits = (float*)calloc(Vp, sizeof(float));
    matvec(logits, m->head->data, xn, Vp, E);
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

            float* x_emb = m->wte->data + generated[g] * E;
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

/* ── Interactive generate ───────────────────────────────────────────────── */

static void generate(CaveModel* m, CaveVocab* vocab, int* prompt, int prompt_len,
                     int max_gen, float temp, float top_p) {
    int Vp = vocab->vocab_size + 2; /* +BOS +MASK */
    int ctx[MAX_SEQ];
    int gen_tokens[MAX_SEQ];
    int ctx_len = 0, gen_len = 0;

    kv_reset();

    /* BOS + prompt */
    ctx[ctx_len++] = vocab->bos_id;
    for (int i = 0; i < prompt_len && ctx_len < CTX; i++)
        ctx[ctx_len++] = prompt[i];

    /* Print prompt glyphs */
    printf("  you: ");
    for (int i = 0; i < prompt_len; i++) printf("%s ", vocab->tokens[prompt[i]]);
    printf("\n  cave: ");

    /* Prefill */
    for (int i = 0; i < ctx_len - 1; i++) {
        float* logits = model_forward(m, ctx[i], i);
        free(logits);
    }

    /* Generate */
    int last_id = ctx[ctx_len - 1];
    for (int step = 0; step < max_gen && ctx_len < CTX; step++) {
        float* logits = model_forward(m, last_id, ctx_len - 1);
        int next = sample_top_p(logits, Vp, temp, top_p);
        free(logits);

        if (next == vocab->bos_id) break;

        ctx[ctx_len++] = next;
        gen_tokens[gen_len++] = next;

        if (next >= 0 && next < vocab->vocab_size)
            printf("%s ", vocab->tokens[next]);
        fflush(stdout);

        last_id = next;
    }
    printf("\n");

    /* ── POST-GENERATION: Hebbian update + co-occurrence + emergence ── */

    /* 1. Hebbian plasticity: learn from this interaction (active, full signal) */
    hebbian_update(m, NULL, vocab->vocab_size + 2, gen_tokens, gen_len, 0);

    /* 2. Count emerged symbol usage */
    count_emerged_usage(m, vocab, gen_tokens, gen_len);

    /* 3. Co-occurrence: track which glyphs appear together */
    int all_tokens[MAX_SEQ];
    int all_len = 0;
    for (int i = 0; i < prompt_len; i++) all_tokens[all_len++] = prompt[i];
    for (int i = 0; i < gen_len; i++) all_tokens[all_len++] = gen_tokens[i];
    cooccur_update(&m->cooccur, all_tokens, all_len);

    /* 3. Decay co-occurrence (slow) */
    cooccur_decay(&m->cooccur, 0.999f);

    /* 5. Check survival — evolution needs death */
    check_symbol_survival(m, vocab);

    /* 6. Try symbol emergence (birth is free, survival is not) */
    int emerged = try_emerge_symbol(m, vocab);
    if (emerged >= 0) {
        printf("  [the cave has a new sign: %s]\n", vocab->tokens[emerged]);
    }
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

/* ── Save/load Hebbian state ────────────────────────────────────────────── */

static void save_state(CaveModel* m, CaveVocab* vocab, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("Cannot save state to %s\n", path); return; }

    /* Co-occurrence */
    fwrite(&m->cooccur, sizeof(CoOccurrence), 1, f);

    /* Emerged symbols */
    fwrite(&m->n_emerged, sizeof(int), 1, f);
    fwrite(m->emerged, sizeof(EmergedSymbol), m->n_emerged, f);

    /* Hebbian LoRA weights */
    for (int l = 0; l < N_L; l++) {
        fwrite(m->layers[l].heb_A_q, sizeof(float), E * HEBBIAN_RANK, f);
        fwrite(m->layers[l].heb_B_q, sizeof(float), HEBBIAN_RANK * E, f);
        fwrite(m->layers[l].heb_A_v, sizeof(float), E * HEBBIAN_RANK, f);
        fwrite(m->layers[l].heb_B_v, sizeof(float), HEBBIAN_RANK * E, f);
    }

    /* Extended vocab */
    fwrite(&vocab->vocab_size, sizeof(int), 1, f);
    for (int i = vocab->base_size; i < vocab->vocab_size; i++)
        fwrite(vocab->tokens[i], 32, 1, f);

    fclose(f);
    printf("  state saved to %s (%d emerged symbols, %d interactions)\n",
           path, m->n_emerged, m->cooccur.total_interactions);
}

static void load_state(CaveModel* m, CaveVocab* vocab, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return; /* first run, no state yet */

    fread(&m->cooccur, sizeof(CoOccurrence), 1, f);
    fread(&m->n_emerged, sizeof(int), 1, f);
    fread(m->emerged, sizeof(EmergedSymbol), m->n_emerged, f);

    for (int l = 0; l < N_L; l++) {
        fread(m->layers[l].heb_A_q, sizeof(float), E * HEBBIAN_RANK, f);
        fread(m->layers[l].heb_B_q, sizeof(float), HEBBIAN_RANK * E, f);
        fread(m->layers[l].heb_A_v, sizeof(float), E * HEBBIAN_RANK, f);
        fread(m->layers[l].heb_B_v, sizeof(float), HEBBIAN_RANK * E, f);
    }

    int saved_vocab;
    fread(&saved_vocab, sizeof(int), 1, f);
    for (int i = vocab->base_size; i < saved_vocab && i < MAX_VOCAB; i++) {
        fread(vocab->tokens[i], 32, 1, f);
        vocab->vocab_size = i + 1;
    }

    fclose(f);
    printf("  state loaded: %d emerged symbols, %d interactions\n",
           m->n_emerged, m->cooccur.total_interactions);
}

/* ── Semantic tokenizer (C version, simplified) ─────────────────────────── */
/* Maps common English words to glyph IDs. Subset of JS semantic_tokenizer. */

typedef struct { const char* word; const char* glyph; } WordMap;

static const WordMap WORD_MAP[] = {
    /* nature */
    {"sun","light"},{"sunrise","light"},{"dawn","light"},{"morning","light"},{"bright","light"},{"shine","light"},
    {"night","dark"},{"shadow","dark"},{"darkness","dark"},{"evening","dark"},{"midnight","dark"},
    {"rain","water"},{"river","water"},{"sea","water"},{"ocean","water"},{"lake","water"},{"swim","water"},
    {"fire","fire"},{"flame","fire"},{"burn","fire"},{"cook","fire"},{"hot","fire"},{"warm","fire"},
    {"ground","earth"},{"soil","earth"},{"land","earth"},{"field","earth"},{"garden","earth"},{"farm","earth"},
    {"rock","stone"},{"mountain","stone"},{"hill","stone"},{"castle","stone"},{"wall","stone"},{"building","stone"},
    {"tree","tree"},{"forest","tree"},{"wood","tree"},{"leaf","tree"},{"flower","tree"},{"grass","tree"},
    {"sky","sky"},{"cloud","sky"},{"wind","sky"},{"storm","sky"},{"air","sky"},
    {"cold","cold"},{"ice","cold"},{"snow","cold"},{"frost","cold"},{"winter","cold"},{"freeze","cold"},
    /* beings */
    {"people","person"},{"human","person"},{"someone","person"},{"everyone","person"},{"they","person"},
    {"he","man"},{"him","man"},{"boy","man"},{"guy","man"},{"father","man"},{"dad","man"},{"husband","man"},{"brother","man"},{"son","man"},{"king","man"},
    {"she","woman"},{"her","woman"},{"girl","woman"},{"mother","woman"},{"mom","woman"},{"wife","woman"},{"sister","woman"},{"daughter","woman"},{"queen","woman"},
    {"child","child"},{"kid","child"},{"baby","child"},{"children","child"},{"kids","child"},{"young","child"},{"little","child"},
    {"old","old"},{"elderly","old"},{"ancient","old"},{"grandfather","old"},{"grandmother","old"},{"grandpa","old"},{"grandma","old"},
    {"god","spirit"},{"prayer","spirit"},{"church","spirit"},{"soul","spirit"},{"angel","spirit"},{"holy","spirit"},
    {"computer","AI"},{"robot","AI"},{"machine","AI"},{"software","AI"},{"technology","AI"},{"digital","AI"},
    {"dog","animal"},{"cat","animal"},{"bird","animal"},{"horse","animal"},{"fish","animal"},{"chicken","animal"},{"rooster","animal"},
    /* body */
    {"hand","body"},{"head","body"},{"face","body"},{"heart","body"},{"eye","body"},{"arm","body"},
    {"eat","food"},{"meal","food"},{"bread","food"},{"coffee","food"},{"tea","food"},{"cake","food"},{"soup","food"},{"beer","food"},{"wine","food"},{"hungry","food"},{"dinner","food"},{"breakfast","food"},{"lunch","food"},
    {"sleep","sleep"},{"bed","sleep"},{"rest","sleep"},{"nap","sleep"},{"pillow","sleep"},{"awake","sleep"},{"wake","sleep"},
    {"hurt","pain"},{"sick","pain"},{"doctor","pain"},{"hospital","pain"},{"medicine","pain"},{"wound","pain"},{"fever","pain"},
    {"strong","strength"},{"power","strength"},{"run","strength"},{"exercise","strength"},{"fight","strength"},{"sport","strength"},
    /* emotion */
    {"happy","joy"},{"smile","joy"},{"laugh","joy"},{"celebrate","joy"},{"dance","joy"},{"fun","joy"},{"enjoy","joy"},
    {"sad","grief"},{"cry","grief"},{"mourn","grief"},{"sorrow","grief"},{"funeral","grief"},{"tears","grief"},
    {"love","love"},{"kiss","love"},{"hug","love"},{"romance","love"},{"wedding","love"},{"marry","love"},
    {"afraid","fear"},{"scared","fear"},{"panic","fear"},{"worry","fear"},{"nightmare","fear"},{"danger","fear"},
    {"angry","anger"},{"mad","anger"},{"rage","anger"},{"hate","anger"},{"yell","anger"},{"shout","anger"},
    {"miss","longing"},{"yearn","longing"},{"homesick","longing"},{"nostalgia","longing"},
    {"tired","tired"},{"exhausted","tired"},{"weary","tired"},{"sleepy","tired"},{"bored","tired"},
    {"stress","stress"},{"pressure","stress"},{"overwhelm","stress"},{"busy","stress"},{"rush","stress"},
    /* verbs */
    {"go","go"},{"walk","go"},{"move","go"},{"travel","go"},{"drive","go"},{"leave","go"},{"arrive","go"},{"come","go"},{"ran","go"},{"went","go"},{"walked","go"},
    {"make","make"},{"build","make"},{"create","make"},{"produce","make"},{"craft","make"},
    {"break","break"},{"destroy","break"},{"smash","break"},{"crash","break"},{"tear","break"},
    {"see","see"},{"look","see"},{"watch","see"},{"read","see"},{"notice","see"},{"found","see"},{"saw","see"},
    {"speak","speak"},{"say","speak"},{"tell","speak"},{"talk","speak"},{"call","speak"},{"sing","speak"},{"said","speak"},{"told","speak"},
    {"hear","hear"},{"listen","hear"},{"sound","hear"},{"music","hear"},{"song","hear"},
    {"seek","seek"},{"search","seek"},{"hunt","seek"},{"explore","seek"},
    {"give","give"},{"share","give"},{"offer","give"},{"send","give"},{"gave","give"},
    {"want","want"},{"wish","want"},{"desire","want"},{"need","want"},{"hope","want"},
    {"miss","miss"},{"lost","miss"},{"gone","miss"},{"absent","miss"},{"lonely","miss"},
    {"agree","agree"},{"yes","agree"},{"accept","agree"},{"nod","agree"},{"peace","agree"},
    /* social */
    {"home","home"},{"house","home"},{"room","home"},{"door","home"},{"kitchen","home"},{"window","home"},{"roof","home"},
    {"outside","outside"},{"nature","outside"},{"park","outside"},{"beach","outside"},{"city","outside"},{"market","outside"},{"shop","outside"},{"street","outside"},
    {"work","work"},{"job","work"},{"office","work"},{"business","work"},{"career","work"},
    {"internet","internet"},{"online","internet"},{"email","internet"},{"phone","internet"},{"website","internet"},
    {"friend","bond"},{"family","bond"},{"together","bond"},{"team","bond"},{"community","bond"},
    {"war","conflict"},{"battle","conflict"},{"attack","conflict"},{"argue","conflict"},{"enemy","conflict"},
    /* mind */
    {"know","know"},{"learn","know"},{"study","know"},{"school","know"},{"book","know"},{"understand","know"},{"knew","know"},{"taught","know"},
    {"idea","idea"},{"plan","idea"},{"concept","idea"},{"solution","idea"},{"invention","idea"},
    {"think","think"},{"thought","think"},{"consider","think"},{"wonder","think"},{"mind","think"},{"decide","think"},
    {"dream","dream"},{"imagine","dream"},{"fantasy","dream"},{"story","dream"},{"wish","dream"},
    {"remember","remember"},{"memory","remember"},{"past","remember"},{"history","remember"},{"forgot","remember"},
    {"lie","lie"},{"cheat","lie"},{"fake","lie"},{"trick","lie"},{"pretend","lie"},
    /* space */
    {"road","path"},{"street","path"},{"way","path"},{"direction","path"},{"trail","path"},
    {"up","up"},{"rise","up"},{"climb","up"},{"above","up"},{"high","up"},{"tall","up"},{"top","up"},
    {"down","down"},{"fall","down"},{"drop","down"},{"below","down"},{"low","down"},{"fell","down"},
    {"far","far"},{"distant","far"},{"away","far"},{"abroad","far"},{"remote","far"},
    {"back","back"},{"return","back"},{"behind","back"},{"again","back"},
    /* time */
    {"before","before"},{"earlier","before"},{"yesterday","before"},{"once","before"},{"ago","before"},
    {"now","now"},{"today","now"},{"moment","now"},{"current","now"},
    {"after","after"},{"later","after"},{"tomorrow","after"},{"soon","after"},{"next","after"},{"then","after"},
    {"never","never"},{"no","never"},{"nothing","never"},{"nobody","never"},{"stop","never"},
    {"always","always"},{"forever","always"},{"every","always"},{"daily","always"},{"constant","always"},
    /* grammar */
    {"not","not"},{"don't","not"},{"can't","not"},{"won't","not"},{"bad","not"},{"wrong","not"},
    {"many","many"},{"much","many"},{"lots","many"},{"several","many"},{"huge","many"},{"thousand","many"},
    {"very","much"},{"really","much"},{"extremely","much"},{"quite","much"},
    {"and","and"},{"also","and"},{"with","and"},{"both","and"},{"plus","and"},
    {"one","one"},{"single","one"},{"alone","one"},{"only","one"},{"first","one"},
    {"question","question"},{"ask","question"},{"why","question"},{"what","question"},{"curious","question"},
    {"how","how"},{"method","how"},{"way","how"},{"step","how"},
    {"because","cause"},{"reason","cause"},{"therefore","cause"},{"result","cause"},
    /* extended */
    {"i","me"},{"my","me"},{"myself","me"},
    {"you","you"},{"your","you"},{"yourself","you"},
    {"other","other"},{"another","other"},{"different","other"},{"new","other"},{"strange","other"},
    {"money","money"},{"dollar","money"},{"pay","money"},{"buy","money"},{"sell","money"},{"rich","money"},{"poor","money"},{"price","money"},
    {"change","change"},{"transform","change"},{"grow","change"},{"develop","change"},{"evolve","change"},
    {"write","write"},{"pen","write"},{"paper","write"},{"letter","write"},{"note","write"},{"wrote","write"},{"poem","write"},{"code","write"},
    {"choose","choose"},{"pick","choose"},{"decide","choose"},{"select","choose"},{"vote","choose"},
    {"help","help"},{"assist","help"},{"support","help"},{"save","help"},{"protect","help"},
    {"have","have"},{"own","have"},{"keep","have"},{"hold","have"},{"got","have"},{"had","have"},
    {"free","free"},{"freedom","free"},{"liberty","free"},{"escape","free"},{"open","free"},
    {"death","death"},{"die","death"},{"dead","death"},{"kill","death"},{"grave","death"},{"died","death"},
    {"music","music"},{"song","music"},{"melody","music"},{"guitar","music"},{"piano","music"},{"drum","music"},{"sang","music"},{"singing","music"},
    {"good","good"},{"great","good"},{"nice","good"},{"kind","good"},{"beautiful","good"},{"wonderful","good"},{"fine","good"},
    /* super */
    {"small","small"},{"tiny","small"},{"little","small"},{"short","small"},{"few","small"},
    {"same","same"},{"equal","same"},{"similar","same"},{"identical","same"},
    {"is","BE"},{"am","BE"},{"are","BE"},{"was","BE"},{"were","BE"},{"being","BE"},{"become","BE"},{"feel","BE"},
    {"wait","wait"},{"patience","wait"},{"pause","wait"},{"delay","wait"},{"stay","wait"},
    {NULL, NULL}
};

static const char* STOP_WORDS[] = {
    "the","a","an","to","of","in","for","on","at","by","from","about","into",
    "through","during","above","between","out","off","over","under","again",
    "further","here","there","when","where","all","each","both","few","more",
    "most","some","such","so","than","too","just","but","if","or","while","as",
    "until","that","this","these","those","it","its","itself","which","who","whom",
    NULL
};

static int is_stop_word(const char* w) {
    for (int i = 0; STOP_WORDS[i]; i++)
        if (strcmp(w, STOP_WORDS[i]) == 0) return 1;
    return 0;
}

static int semantic_tokenize_word(const char* word, CaveVocab* vocab) {
    /* Direct glyph name match first */
    int id = vocab_find(vocab, word);
    if (id >= 0) return id;

    /* Word map lookup */
    for (int i = 0; WORD_MAP[i].word; i++) {
        if (strcmp(word, WORD_MAP[i].word) == 0) {
            return vocab_find(vocab, WORD_MAP[i].glyph);
        }
    }
    return -1;
}

static int semantic_tokenize_line(const char* line, CaveVocab* vocab, int* out, int max_tokens) {
    char buf[4096];
    strncpy(buf, line, 4095); buf[4095] = '\0';

    /* Lowercase */
    for (int i = 0; buf[i]; i++)
        if (buf[i] >= 'A' && buf[i] <= 'Z') buf[i] += 32;

    /* Strip punctuation */
    for (int i = 0; buf[i]; i++)
        if (!((buf[i] >= 'a' && buf[i] <= 'z') || (buf[i] >= '0' && buf[i] <= '9') ||
              buf[i] == ' ' || buf[i] == '\'' || buf[i] == '-'))
            buf[i] = ' ';

    int n = 0, last_id = -1;
    char* tok = strtok(buf, " \t\n");
    while (tok && n < max_tokens) {
        if (strlen(tok) == 0 || is_stop_word(tok)) { tok = strtok(NULL, " \t\n"); continue; }
        int id = semantic_tokenize_word(tok, vocab);
        if (id >= 0 && id != last_id) {
            out[n++] = id;
            last_id = id;
        }
        tok = strtok(NULL, " \t\n");
    }
    return n;
}

/* ── Async self-learning thread ─────────────────────────────────────────── */

typedef struct {
    CaveModel*  model;
    CaveVocab*  vocab;
    const char* feed_dir;    /* directory to watch for .txt files */
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
        int n = semantic_tokenize_line(sentence, al->vocab, tokens, MAX_SEQ - 1);
        if (n >= 2) {
            pthread_mutex_lock(&al->lock);

            /* Hebbian: passive reading — 0.3x signal, V-only */
            hebbian_update(al->model, NULL, al->vocab->vocab_size + 2, tokens, n, 1);

            /* Co-occurrence */
            cooccur_update(&al->model->cooccur, tokens, n);

            /* Survival check + emergence */
            check_symbol_survival(al->model, al->vocab);
            try_emerge_symbol(al->model, al->vocab);

            al->lines_learned++;
            pthread_mutex_unlock(&al->lock);
        }
    }
    free(buf);
}

static void* learner_thread(void* arg) {
    AsyncLearner* al = (AsyncLearner*)arg;

    while (al->running) {
        DIR* dir = opendir(al->feed_dir);
        if (!dir) { sleep(5); continue; }

        struct dirent* ent;
        while ((ent = readdir(dir)) != NULL) {
            if (!al->running) break;

            /* Only .txt files */
            int nlen = (int)strlen(ent->d_name);
            if (nlen < 5 || strcmp(ent->d_name + nlen - 4, ".txt") != 0) continue;

            /* Build path */
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", al->feed_dir, ent->d_name);

            /* Check if already consumed (rename to .learned) */
            char done_path[1024];
            snprintf(done_path, sizeof(done_path), "%s/%s.learned", al->feed_dir, ent->d_name);
            struct stat st;
            if (stat(done_path, &st) == 0) continue; /* already processed */

            /* Read file */
            FILE* f = fopen(path, "r");
            if (!f) continue;
            fseek(f, 0, SEEK_END);
            long fsize = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (fsize > 2 * 1024 * 1024) { fclose(f); continue; } /* max 2MB per file */
            char* content = (char*)malloc(fsize + 1);
            fread(content, 1, fsize, f);
            content[fsize] = '\0';
            fclose(f);

            printf("\n  [learner] consuming %s (%ld bytes)...\n", ent->d_name, fsize);
            learn_from_text(al, content, (int)fsize);
            free(content);

            /* Mark as consumed */
            rename(path, done_path);
            al->files_consumed++;
            printf("  [learner] %s → %d lines learned (total: %d)\n",
                   ent->d_name, al->lines_learned, al->files_consumed);
        }
        closedir(dir);

        /* Sleep 10 seconds before next scan */
        for (int i = 0; i < 10 && al->running; i++) sleep(1);
    }
    return NULL;
}

static AsyncLearner g_learner;
static pthread_t    g_learner_tid;

static void start_learner(CaveModel* model, CaveVocab* vocab, const char* feed_dir) {
    /* Create feed directory if needed */
    mkdir(feed_dir, 0755);

    g_learner.model = model;
    g_learner.vocab = vocab;
    g_learner.feed_dir = feed_dir;
    g_learner.running = 1;
    g_learner.files_consumed = 0;
    g_learner.lines_learned = 0;
    pthread_mutex_init(&g_learner.lock, NULL);

    pthread_create(&g_learner_tid, NULL, learner_thread, &g_learner);
    printf("  [learner] watching %s/ for .txt files\n", feed_dir);
}

static void stop_learner(void) {
    g_learner.running = 0;
    pthread_join(g_learner_tid, NULL);
    pthread_mutex_destroy(&g_learner.lock);
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    const char* weights_path = "weights/cavellman_v3.bin";
    const char* state_path = "weights/cavellman.state";
    const char* preset_name = "small";
    float temp = 0.8f, top_p = 0.9f;
    int max_gen = 0, seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--weights") == 0 && i+1 < argc) weights_path = argv[++i];
        else if (strcmp(argv[i], "--state") == 0 && i+1 < argc) state_path = argv[++i];
        else if (strcmp(argv[i], "--preset") == 0 && i+1 < argc) preset_name = argv[++i];
        else if (strcmp(argv[i], "--temp") == 0 && i+1 < argc) temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i+1 < argc) top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--max") == 0 && i+1 < argc) max_gen = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("cavellman — self-evolving hieroglyphic language model\n\n");
            printf("  --weights FILE   Weight file (default: weights/cavellman_v3.bin)\n");
            printf("  --state FILE     Hebbian state file (default: weights/cavellman.state)\n");
            printf("  --preset NAME    Model preset (default: small)\n");
            printf("  --temp FLOAT     Temperature (default: 0.8)\n");
            printf("  --top-p FLOAT    Nucleus sampling (default: 0.9)\n");
            printf("  --max N          Max tokens to generate (default: CTX)\n");
            printf("  --seed N         RNG seed (default: 42)\n");
            return 0;
        }
    }

    /* Apply preset */
    const Preset* pr = NULL;
    for (int i = 0; i < N_PRESETS; i++)
        if (strcmp(PRESETS[i].name, preset_name) == 0) { pr = &PRESETS[i]; break; }
    if (!pr) { printf("Unknown preset: %s\n", preset_name); return 1; }
    E = pr->embd; H = pr->heads; HD = E / H; FFN_D = 4 * E; N_L = pr->layers; CTX = pr->ctx;
    if (max_gen <= 0) max_gen = CTX - 1;
    srand(seed);

    printf("══════════════════════════════════════════════════════════\n");
    printf("  caveLLMan — self-evolving hieroglyphic language model\n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  weights: %s\n  preset: %s (E=%d H=%d L=%d CTX=%d)\n",
           weights_path, preset_name, E, H, N_L, CTX);
    printf("  temp=%.2f, top_p=%.2f, max=%d\n", temp, top_p, max_gen);
    printf("  hebbian: rank=%d, lr=%.4f\n", HEBBIAN_RANK, 0.001f);
    printf("  emergence: threshold=%.2f, window=%d\n", EMERGE_THRESHOLD, DISCIPLINE_WINDOW);

    /* Load vocab */
    char vocab_path[512];
    snprintf(vocab_path, sizeof(vocab_path), "%s.vocab", weights_path);
    CaveVocab vocab;
    memset(&vocab, 0, sizeof(vocab));
    if (load_vocab(vocab_path, &vocab) != 0) return 1;
    printf("  vocab: %d base glyphs\n", vocab.vocab_size);

    /* Load model */
    printf("  loading weights...\n");
    nt_seed(seed);
    CaveModel* model = model_load(weights_path, vocab.vocab_size);
    if (!model) return 1;

    /* Load Hebbian state (if exists) */
    load_state(model, &vocab, state_path);
    if (model->n_emerged > 0) {
        printf("  emerged symbols: ");
        for (int i = 0; i < model->n_emerged; i++)
            printf("%s ", model->emerged[i].name);
        printf("\n");
    }

    printf("══════════════════════════════════════════════════════════\n\n");
    /* Start async learner (watches feed/ directory) */
    start_learner(model, &vocab, "feed");

    printf("Speak in glyphs. Type 'quit' to exit. '?' for glyph list.\n");
    printf("Drop .txt files into feed/ — the cave devours them.\n\n");

    /* Interactive loop */
    char input[1024];
    while (1) {
        printf("▸ ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;

        int len = (int)strlen(input);
        while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r')) input[--len] = '\0';
        if (len == 0) continue;
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) break;
        if (strcmp(input, "?") == 0) {
            printf("\n  Base glyphs (%d):\n  ", vocab.base_size);
            for (int i = 0; i < vocab.base_size; i++) {
                printf("%s ", vocab.tokens[i]);
                if ((i + 1) % 12 == 0) printf("\n  ");
            }
            if (model->n_emerged > 0) {
                printf("\n\n  Emerged symbols (%d):\n  ", model->n_emerged);
                for (int i = 0; i < model->n_emerged; i++)
                    printf("%s (%.2f) ", model->emerged[i].name,
                           model->emerged[i].strength);
            }
            printf("\n\n");
            continue;
        }
        if (strcmp(input, "save") == 0) {
            save_state(model, &vocab, state_path);
            continue;
        }
        if (strcmp(input, "stats") == 0) {
            printf("  interactions: %d\n", model->cooccur.total_interactions);
            printf("  emerged: %d/%d\n", model->n_emerged, MAX_EMERGED);
            printf("  next emergence window: %d interactions\n",
                   DISCIPLINE_WINDOW - (model->cooccur.total_interactions - model->cooccur.last_emergence));
            printf("  top co-occurrences:\n");
            for (int i = 0; i < vocab.base_size; i++)
                for (int j = i+1; j < vocab.base_size; j++)
                    if (model->cooccur.matrix[i][j] > 0.5f)
                        printf("    %s + %s = %.3f\n", vocab.tokens[i], vocab.tokens[j], model->cooccur.matrix[i][j]);
            printf("\n");
            continue;
        }

        /* Tokenize and generate */
        int prompt_tokens[MAX_SEQ];
        int n_prompt = tokenize_prompt(input, &vocab, prompt_tokens, CTX - 2);
        if (n_prompt == 0) {
            printf("  (no recognized glyphs, type ? for list)\n");
            continue;
        }

        pthread_mutex_lock(&g_learner.lock);
        generate(model, &vocab, prompt_tokens, n_prompt, max_gen, temp, top_p);
        pthread_mutex_unlock(&g_learner.lock);
    }

    /* Stop learner and save */
    stop_learner();
    save_state(model, &vocab, state_path);

    printf("\nthe cave remembers. (%d files consumed, %d lines learned)\n",
           g_learner.files_consumed, g_learner.lines_learned);
    free(model->layers);
    free(model);
    return 0;
}
