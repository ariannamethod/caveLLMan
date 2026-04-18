/*
 * train_emolm.c — Train emoLM transformer on emoji stories using notorch
 *
 * No Python. No pip. No torch. Pure C.
 * Emoji-level tokenizer (space-split), GPT architecture, Chuck optimizer.
 *
 * Build: make train
 * Run:   ./train_emolm [dataset] [steps] [lr]
 *
 * Default: data/emoji_stories.txt, 2000 steps, lr=3e-4
 *
 * Copyright (C) 2026 Arianna Method contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

/* ── Config (tiny preset — fast training, matches emolm.py tiny) ────────── */

#define MAX_VOCAB   256     /* max emoji types */
#define MAX_SEQ     128     /* block_size — covers all presets */
#define MAX_STORIES 512
#define MAX_STORY_LEN 64

/* Model architecture — configurable at runtime */
static int E     = 18;     /* embedding dim */
static int H     = 3;      /* attention heads */
static int HD    = 6;      /* head_dim = E / H */
static int FFN_D = 72;     /* 4 * E */
static int N_L   = 2;      /* layers */
static int CTX   = 32;     /* context window */

/* ── Emoji tokenizer ─────────────────────────────────────────────────────── */

typedef struct {
    char tokens[MAX_VOCAB][32];  /* UTF-8 emoji strings (ZWJ sequences up to 28 bytes) */
    int  vocab_size;             /* actual number of unique emojis */
    int  bos_id;                 /* vocab_size = BOS/END token */
} EmojiVocab;

/* Stories as token ID sequences */
typedef struct {
    int  data[MAX_STORIES][MAX_STORY_LEN];
    int  lens[MAX_STORIES];
    int  count;
} StoryData;

static int vocab_find(EmojiVocab* v, const char* tok) {
    for (int i = 0; i < v->vocab_size; i++) {
        if (strcmp(v->tokens[i], tok) == 0) return i;
    }
    return -1;
}

static int vocab_add(EmojiVocab* v, const char* tok) {
    int id = vocab_find(v, tok);
    if (id >= 0) return id;
    if (v->vocab_size >= MAX_VOCAB - 1) return -1; /* reserve 1 for BOS */
    id = v->vocab_size;
    strncpy(v->tokens[id], tok, 31);
    v->tokens[id][31] = '\0';
    v->vocab_size++;
    return id;
}

/*
 * Read UTF-8 codepoint at *p.
 * Returns number of bytes consumed, writes codepoint to *cp.
 */
static int utf8_next(const char* p, int* cp) {
    unsigned char c = (unsigned char)*p;
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0) { *cp = (c & 0x1F) << 6 | (p[1] & 0x3F); return 2; }
    if ((c & 0xF0) == 0xE0) { *cp = (c & 0x0F) << 12 | (p[1] & 0x3F) << 6 | (p[2] & 0x3F); return 3; }
    if ((c & 0xF8) == 0xF0) { *cp = (c & 0x07) << 18 | (p[1] & 0x3F) << 12 | (p[2] & 0x3F) << 6 | (p[3] & 0x3F); return 4; }
    *cp = c; return 1;
}

/*
 * Load emoji stories from file.
 * Format: one story per line, emojis separated by spaces.
 * Each emoji is a space-delimited token (may be multi-byte UTF-8 + variant selectors).
 */
static int load_stories(const char* path, EmojiVocab* vocab, StoryData* stories) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("Cannot open %s\n", path); return -1; }

    vocab->vocab_size = 0;
    stories->count = 0;

    char line[4096];
    while (fgets(line, sizeof(line), f) && stories->count < MAX_STORIES) {
        /* Strip newline */
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        /* Tokenize by spaces */
        int sid = stories->count;
        stories->lens[sid] = 0;

        char* tok = strtok(line, " ");
        while (tok && stories->lens[sid] < MAX_STORY_LEN - 2) {
            /* Skip empty tokens */
            if (strlen(tok) == 0) { tok = strtok(NULL, " "); continue; }
            int id = vocab_add(vocab, tok);
            if (id >= 0) {
                stories->data[sid][stories->lens[sid]++] = id;
            }
            tok = strtok(NULL, " ");
        }
        if (stories->lens[sid] > 0) stories->count++;
    }
    fclose(f);

    vocab->bos_id = vocab->vocab_size; /* BOS/END = vocab_size */
    return 0;
}

/* ── Model ───────────────────────────────────────────────────────────────── */

typedef struct {
    nt_tensor *wte;     /* [V+1, E] — includes BOS token */
    nt_tensor *wpe;     /* [CTX, E] */
    struct {
        nt_tensor *rms1;       /* [E] — pre-attention norm */
        nt_tensor *wq, *wk, *wv, *wo; /* [E, E] */
        nt_tensor *rms2;       /* [E] — pre-FFN norm */
        nt_tensor *w_fc1;      /* [FFN, E] — SiLU gate */
        nt_tensor *w_fc2;      /* [E, FFN] — down projection */
    } layers[8]; /* max 8 layers */
    nt_tensor *rms_f;   /* [E] — final norm */
    nt_tensor *head;    /* [V+1, E] */
} EmoModel;

static long count_params(EmoModel* m, int V) {
    long n = m->wte->len + m->wpe->len + m->rms_f->len + m->head->len;
    for (int l = 0; l < N_L; l++) {
        n += m->layers[l].rms1->len + m->layers[l].rms2->len;
        n += m->layers[l].wq->len + m->layers[l].wk->len;
        n += m->layers[l].wv->len + m->layers[l].wo->len;
        n += m->layers[l].w_fc1->len + m->layers[l].w_fc2->len;
    }
    (void)V;
    return n;
}

static EmoModel* model_create(int V) {
    /* V = vocab_size + 1 (BOS) */
    int Vp = V + 1;
    EmoModel* m = (EmoModel*)calloc(1, sizeof(EmoModel));

    m->wte = nt_tensor_new2d(Vp, E);
    nt_tensor_xavier(m->wte, Vp, E);
    m->wpe = nt_tensor_new2d(CTX, E);
    nt_tensor_xavier(m->wpe, CTX, E);

    float scale_res = 0.02f / sqrtf(2.0f * N_L);
    for (int l = 0; l < N_L; l++) {
        m->layers[l].rms1 = nt_tensor_new(E);
        nt_tensor_fill(m->layers[l].rms1, 1.0f);
        m->layers[l].wq = nt_tensor_new2d(E, E);
        nt_tensor_xavier(m->layers[l].wq, E, E);
        m->layers[l].wk = nt_tensor_new2d(E, E);
        nt_tensor_xavier(m->layers[l].wk, E, E);
        m->layers[l].wv = nt_tensor_new2d(E, E);
        nt_tensor_xavier(m->layers[l].wv, E, E);
        m->layers[l].wo = nt_tensor_new2d(E, E);
        nt_tensor_xavier(m->layers[l].wo, E, E);
        /* Scale residual output init */
        for (int i = 0; i < m->layers[l].wo->len; i++)
            m->layers[l].wo->data[i] *= scale_res / 0.1f;

        m->layers[l].rms2 = nt_tensor_new(E);
        nt_tensor_fill(m->layers[l].rms2, 1.0f);
        m->layers[l].w_fc1 = nt_tensor_new2d(FFN_D, E);
        nt_tensor_xavier(m->layers[l].w_fc1, E, FFN_D);
        m->layers[l].w_fc2 = nt_tensor_new2d(E, FFN_D);
        nt_tensor_xavier(m->layers[l].w_fc2, FFN_D, E);
        for (int i = 0; i < m->layers[l].w_fc2->len; i++)
            m->layers[l].w_fc2->data[i] *= scale_res / 0.1f;
    }

    m->rms_f = nt_tensor_new(E);
    nt_tensor_fill(m->rms_f, 1.0f);
    m->head = nt_tensor_new2d(Vp, E);
    nt_tensor_xavier(m->head, E, Vp);

    return m;
}

static void model_free(EmoModel* m) {
    nt_tensor_free(m->wte); nt_tensor_free(m->wpe);
    for (int l = 0; l < N_L; l++) {
        nt_tensor_free(m->layers[l].rms1); nt_tensor_free(m->layers[l].rms2);
        nt_tensor_free(m->layers[l].wq); nt_tensor_free(m->layers[l].wk);
        nt_tensor_free(m->layers[l].wv); nt_tensor_free(m->layers[l].wo);
        nt_tensor_free(m->layers[l].w_fc1); nt_tensor_free(m->layers[l].w_fc2);
    }
    nt_tensor_free(m->rms_f); nt_tensor_free(m->head);
    free(m);
}

/* ── Forward pass on tape ────────────────────────────────────────────────── */

static int model_forward(EmoModel* m, int* tokens, int* targets, int seq_len, int V) {
    int Vp = V + 1;

    /* Register params */
    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
    int wpe_i = nt_tape_param(m->wpe); nt_tape_no_decay(wpe_i);

    int li[8][8]; /* layer_idx[layer][param_idx] */
    for (int l = 0; l < N_L; l++) {
        li[l][0] = nt_tape_param(m->layers[l].rms1);
        li[l][1] = nt_tape_param(m->layers[l].wq);
        li[l][2] = nt_tape_param(m->layers[l].wk);
        li[l][3] = nt_tape_param(m->layers[l].wv);
        li[l][4] = nt_tape_param(m->layers[l].wo);
        li[l][5] = nt_tape_param(m->layers[l].rms2);
        li[l][6] = nt_tape_param(m->layers[l].w_fc1);
        li[l][7] = nt_tape_param(m->layers[l].w_fc2);
    }
    int rmsf_i = nt_tape_param(m->rms_f);
    int head_i = nt_tape_param(m->head);

    /* Input tokens as tensor */
    nt_tensor* tok_t = nt_tensor_new(seq_len);
    nt_tensor* tgt_t = nt_tensor_new(seq_len);
    for (int i = 0; i < seq_len; i++) {
        tok_t->data[i] = (float)tokens[i];
        tgt_t->data[i] = (float)targets[i];
    }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t);
    nt_tensor_free(tgt_t);

    /* Embed: h = wte[tokens] + wpe[positions] */
    int h = nt_seq_embedding(wte_i, wpe_i, tok_i, seq_len, E);

    /* Transformer blocks */
    for (int l = 0; l < N_L; l++) {
        /* RMSNorm → Attention */
        int xn = nt_seq_rmsnorm(h, li[l][0], seq_len, E);
        int q = nt_seq_linear(li[l][1], xn, seq_len);
        int k = nt_seq_linear(li[l][2], xn, seq_len);
        int v = nt_seq_linear(li[l][3], xn, seq_len);
        int attn = nt_mh_causal_attention(q, k, v, seq_len, HD);
        int proj = nt_seq_linear(li[l][4], attn, seq_len);
        h = nt_add(h, proj); /* residual */

        /* RMSNorm → FFN (SiLU gate + down) */
        xn = nt_seq_rmsnorm(h, li[l][5], seq_len, E);
        int fc1 = nt_seq_linear(li[l][6], xn, seq_len);
        fc1 = nt_silu(fc1);
        int fc2 = nt_seq_linear(li[l][7], fc1, seq_len);
        h = nt_add(h, fc2); /* residual */
    }

    /* Final norm + head */
    int hf = nt_seq_rmsnorm(h, rmsf_i, seq_len, E);
    int logits = nt_seq_linear(head_i, hf, seq_len);

    /* Loss */
    int loss = nt_seq_cross_entropy(logits, tgt_i, seq_len, Vp);
    return loss;
}

/* ── Timer ───────────────────────────────────────────────────────────────── */

static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ── Presets ──────────────────────────────────────────────────────────────── */

typedef struct {
    const char* name;
    int embd, heads, layers, ctx, steps;
} Preset;

static const Preset PRESETS[] = {
    {"tiny",     18, 3, 2, 32,  2000},
    {"micro",    48, 4, 3, 64,  3000},
    {"standard", 64, 8, 3, 64,  4000},
    {"small",    96, 8, 4, 128, 5000},
    {"medium",  128, 8, 4, 128, 8000},
};
#define N_PRESETS 5

static const Preset* find_preset(const char* name) {
    for (int i = 0; i < N_PRESETS; i++) {
        if (strcmp(PRESETS[i].name, name) == 0) return &PRESETS[i];
    }
    return NULL;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    const char* dataset = "data/emoji_stories.txt";
    const char* preset_name = "tiny";
    int    steps = 0; /* 0 = use preset default */
    float  base_lr = 3e-4f;
    const char* save_path = "weights/emolm.bin";  /* save by default */
    int    no_save = 0;
    int    seed = 42;

    /* Simple arg parsing */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) {
            dataset = argv[++i];
        } else if (strcmp(argv[i], "--preset") == 0 && i + 1 < argc) {
            preset_name = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            base_lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else if (strcmp(argv[i], "--no-save") == 0) {
            no_save = 1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("train_emolm — emoLM training on notorch (pure C)\n\n");
            printf("  --dataset FILE    Dataset (default: data/emoji_stories.txt)\n");
            printf("  --preset NAME     tiny/micro/standard/small/medium (default: tiny)\n");
            printf("  --steps N         Training steps (default: preset)\n");
            printf("  --lr FLOAT        Learning rate (default: 3e-4)\n");
            printf("  --save FILE       Save weights (default: weights/emolm.bin)\n");
            printf("  --no-save         Don't save weights\n");
            printf("  --seed N          RNG seed (default: 42)\n");
            return 0;
        }
    }

    /* Apply preset */
    const Preset* pr = find_preset(preset_name);
    if (!pr) {
        printf("Unknown preset: %s\n", preset_name);
        printf("Available: tiny, micro, standard, small, medium\n");
        return 1;
    }
    E     = pr->embd;
    H     = pr->heads;
    HD    = E / H;
    FFN_D = 4 * E;
    N_L   = pr->layers;
    CTX   = pr->ctx;
    if (steps <= 0) steps = pr->steps;

    printf("════════════════════════════════════════════════════════\n");
    printf("  emoLM — Emoji Story GPT on notorch (pure C)\n");
    printf("════════════════════════════════════════════════════════\n");
    printf("  dataset: %s\n", dataset);
    printf("  preset:  %s (E=%d H=%d L=%d CTX=%d)\n", preset_name, E, H, N_L, CTX);
    printf("  steps:   %d, lr=%.1e, seed=%d\n", steps, base_lr, seed);

    /* Load data */
    EmojiVocab vocab;
    StoryData stories;
    memset(&vocab, 0, sizeof(vocab));
    memset(&stories, 0, sizeof(stories));

    if (load_stories(dataset, &vocab, &stories) != 0) return 1;
    int V = vocab.vocab_size;
    printf("  stories: %d, vocab: %d emojis + ⏹end = %d\n",
           stories.count, V, V + 1);

    /* Print vocab sample */
    printf("  vocab:   ");
    int show = V < 20 ? V : 20;
    for (int i = 0; i < show; i++) printf("%s ", vocab.tokens[i]);
    if (V > 20) printf("...");
    printf("\n");

    /* Create model */
    nt_seed(seed);
    EmoModel* model = model_create(V);
    long np = count_params(model, V);
    printf("  params:  %ld (%.1f KB)\n", np, np * 4.0f / 1024.0f);
    printf("════════════════════════════════════════════════════════\n\n");

    /* LR schedule: cosine with warmup */
    nt_schedule sched = nt_schedule_cosine(base_lr, steps / 10, steps, base_lr * 0.1f);
    nt_nan_guard guard = nt_nan_guard_new();

    /* Training loop */
    printf("training...\n");
    printf("──────────────────────────────────────────────────\n");

    double t0 = now_ms();
    float first_loss = 0, last_loss = 0;
    int log_every = steps > 200 ? steps / 50 : 4;
    if (log_every < 1) log_every = 1;

    for (int step = 0; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);

        /* Pick story (round-robin) */
        int sid = step % stories.count;
        int slen = stories.lens[sid];

        /* Build tokens: BOS + story + BOS, capped to CTX */
        int tokens[MAX_SEQ + 2], targets[MAX_SEQ + 2];
        int n = slen + 1; /* +1 for BOS prefix */
        if (n > CTX) n = CTX;

        tokens[0] = vocab.bos_id;
        for (int i = 1; i < n; i++) {
            tokens[i] = (i - 1 < slen) ? stories.data[sid][i - 1] : vocab.bos_id;
        }
        for (int i = 0; i < n - 1; i++) {
            targets[i] = tokens[i + 1];
        }
        targets[n - 1] = vocab.bos_id; /* predict END */
        int seq = n;

        /* Forward */
        nt_tape_start();
        int loss_idx = model_forward(model, tokens, targets, seq, V);
        float loss_val = nt_tape_get()->entries[loss_idx].output->data[0];

        if (step == 0) first_loss = loss_val;
        last_loss = loss_val;

        /* Backward */
        nt_tape_backward(loss_idx);

        /* NaN check */
        if (!nt_nan_guard_check(&guard)) {
            nt_tape_clear();
            continue;
        }

        /* Gradient clip + Chuck step */
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, loss_val);
        nt_tape_clear();

        /* Log */
        if ((step + 1) % log_every == 0 || step == 0 || step == steps - 1) {
            double elapsed = (now_ms() - t0) / 1000.0;
            printf("  step %4d/%d | loss %.4f | lr %.2e | %.1fs\n",
                   step + 1, steps, loss_val, lr, elapsed);
            fflush(stdout);
        }
    }

    double total_s = (now_ms() - t0) / 1000.0;
    printf("──────────────────────────────────────────────────\n");
    printf("  loss: %.4f → %.4f", first_loss, last_loss);
    if (first_loss > 0)
        printf(" (%.1f%% reduction)", (first_loss - last_loss) / first_loss * 100.0f);
    printf("\n");
    printf("  time: %.1f seconds (%.1f steps/s)\n", total_s, steps / total_s);
    printf("  nans: %d detected, %d skipped\n", guard.total_nan_count, guard.skipped_steps);
    printf("✅ Training complete\n\n");

    /* ── Generation ─────────────────────────────────────────────────────── */
    printf("── sample stories ──\n");
    nt_train_mode(0); /* eval mode */

    int Vp = V + 1;
    for (int s = 0; s < 8; s++) {
        /* Start with BOS */
        int ctx[MAX_SEQ];
        ctx[0] = vocab.bos_id;
        int gen_len = 1;

        for (int step = 0; step < CTX - 1 && gen_len < CTX; step++) {
            nt_tape_start();

            /* Build token/target arrays for current sequence */
            int toks[MAX_SEQ], tgts[MAX_SEQ];
            for (int i = 0; i < gen_len; i++) toks[i] = ctx[i];
            for (int i = gen_len; i < CTX; i++) toks[i] = 0;
            for (int i = 0; i < CTX; i++) tgts[i] = 0;

            int loss_idx_g = model_forward(model, toks, tgts, gen_len, V);

            /* Get logits for last position */
            nt_tape* tape = nt_tape_get();
            /* logits parent of cross_entropy is parent1 */
            int logits_idx = tape->entries[loss_idx_g].parent1;
            nt_tensor* logits = tape->entries[logits_idx].output;

            /* Sample from last position */
            float* last_logits = logits->data + (gen_len - 1) * Vp;
            float temp = 0.8f;

            /* Temperature + softmax */
            float mx = last_logits[0];
            for (int i = 1; i < Vp; i++) if (last_logits[i] > mx) mx = last_logits[i];
            float sum = 0;
            float probs[MAX_VOCAB + 1];
            for (int i = 0; i < Vp; i++) {
                probs[i] = expf((last_logits[i] - mx) / temp);
                sum += probs[i];
            }
            for (int i = 0; i < Vp; i++) probs[i] /= sum;

            /* Top-p / nucleus sampling */
            float top_p = 0.9f;
            /* Sort indices by probability descending (insertion sort, Vp is small) */
            int indices[MAX_VOCAB + 1];
            for (int i = 0; i < Vp; i++) indices[i] = i;
            for (int i = 1; i < Vp; i++) {
                int key = indices[i];
                float kp = probs[key];
                int j = i - 1;
                while (j >= 0 && probs[indices[j]] < kp) {
                    indices[j + 1] = indices[j]; j--;
                }
                indices[j + 1] = key;
            }
            /* Accumulate until we reach top_p */
            float cum_p = 0;
            int nucleus_size = 0;
            for (int i = 0; i < Vp; i++) {
                cum_p += probs[indices[i]];
                nucleus_size++;
                if (cum_p >= top_p) break;
            }
            /* Renormalize nucleus */
            float nuc_sum = 0;
            for (int i = 0; i < nucleus_size; i++) nuc_sum += probs[indices[i]];
            /* Sample from nucleus */
            float r = (float)rand() / (float)RAND_MAX;
            float cum = 0;
            int next = vocab.bos_id;
            for (int i = 0; i < nucleus_size; i++) {
                cum += probs[indices[i]] / nuc_sum;
                if (cum >= r) { next = indices[i]; break; }
            }

            nt_tape_clear();

            if (next == vocab.bos_id) break; /* END token */
            ctx[gen_len++] = next;
        }

        /* Print story */
        printf("  %d: ", s + 1);
        for (int i = 1; i < gen_len; i++) { /* skip BOS */
            if (ctx[i] >= 0 && ctx[i] < V)
                printf("%s ", vocab.tokens[ctx[i]]);
        }
        printf("\n");
    }

    /* ── Save weights ──────────────────────────────────────────────────── */
    if (save_path && !no_save) {
        printf("\n── saving weights ──\n");
        /* Collect all param tensors */
        nt_tensor* params[256];
        int pi = 0;
        params[pi++] = model->wte;
        params[pi++] = model->wpe;
        for (int l = 0; l < N_L; l++) {
            params[pi++] = model->layers[l].rms1;
            params[pi++] = model->layers[l].wq;
            params[pi++] = model->layers[l].wk;
            params[pi++] = model->layers[l].wv;
            params[pi++] = model->layers[l].wo;
            params[pi++] = model->layers[l].rms2;
            params[pi++] = model->layers[l].w_fc1;
            params[pi++] = model->layers[l].w_fc2;
        }
        params[pi++] = model->rms_f;
        params[pi++] = model->head;
        nt_save(save_path, params, pi);
        printf("  saved %d tensors to %s\n", pi, save_path);

        /* Save metadata JSON alongside weights */
        char meta_path[512];
        snprintf(meta_path, sizeof(meta_path), "%s.json", save_path);
        FILE* mf = fopen(meta_path, "w");
        if (mf) {
            fprintf(mf, "{\n");
            fprintf(mf, "  \"preset\": \"%s\",\n", preset_name);
            fprintf(mf, "  \"embd\": %d,\n", E);
            fprintf(mf, "  \"heads\": %d,\n", H);
            fprintf(mf, "  \"layers\": %d,\n", N_L);
            fprintf(mf, "  \"ctx\": %d,\n", CTX);
            fprintf(mf, "  \"vocab_size\": %d,\n", V);
            fprintf(mf, "  \"params\": %ld,\n", np);
            fprintf(mf, "  \"steps\": %d,\n", steps);
            fprintf(mf, "  \"final_loss\": %.4f,\n", last_loss);
            fprintf(mf, "  \"dataset\": \"%s\",\n", dataset);
            fprintf(mf, "  \"seed\": %d\n", seed);
            fprintf(mf, "}\n");
            fclose(mf);
            printf("  metadata: %s\n", meta_path);
        }

        /* Save vocab for inferencer */
        char vocab_path[512];
        snprintf(vocab_path, sizeof(vocab_path), "%s.vocab", save_path);
        FILE* vf = fopen(vocab_path, "w");
        if (vf) {
            fprintf(vf, "%d\n", V);
            for (int i = 0; i < V; i++)
                fprintf(vf, "%s\n", vocab.tokens[i]);
            fclose(vf);
            printf("  vocab: %s (%d tokens)\n", vocab_path, V);
        }
    }

    model_free(model);

    printf("\n════════════════════════════════════════════════════════\n");
    printf("  No Python was harmed. fuck torch.\n");
    printf("════════════════════════════════════════════════════════\n");
    return 0;
}
