/*
 * train_cavellman.c — Train caveLLMan transformer on 88-glyph sequences
 *
 * No Python. No pip. No torch. Pure C + notorch.
 * Glyph-level tokenizer (space-split), GPT architecture, Chuck optimizer.
 *
 * Build: make train
 * Run:   ./train_cavellman [--dataset FILE] [--preset NAME] [--steps N]
 *
 * Default: data/cavellman_train_final.txt, small preset, cosine LR.
 *
 * Copyright (C) 2026 Arianna Method contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "notorch.h"
#include "semantic_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* ── Config ─────────────────────────────────────────────────────────────── */

#define MAX_VOCAB   256     /* 88 base glyphs + room for emerged symbols */
#define MAX_SEQ     128     /* block_size — covers all presets */
#define MAX_STORIES 100000  /* raised for big corpora (~13MB fineweb+classics) */
#define MAX_STORY_LEN 64

/* Model architecture — configurable at runtime */
static int E     = 96;     /* embedding dim */
static int H     = 8;      /* attention heads */
static int HD    = 12;     /* head_dim = E / H */
static int FFN_D = 384;    /* 4 * E */
static int N_L   = 4;      /* layers */
static int CTX   = 128;    /* context window */

/* ── Glyph vocab + story container ─────────────────────────────────────── */

typedef struct {
    char tokens[MAX_VOCAB][32];  /* glyph names seeded from semantic_tokenizer.h */
    int  vocab_size;             /* = GLYPH_COUNT (88) after seeding */
    int  bos_id;                 /* = GLYPH_COUNT */
} GlyphVocab;

typedef struct {
    int  data[MAX_STORIES][MAX_STORY_LEN];
    int  lens[MAX_STORIES];
    int  count;
} StoryData;

/*
 * Load raw English from `path`, split into sentences on .!? boundaries
 * (SPA phonon style, same logic the engine uses in learn_from_text),
 * then compress each sentence through semtok_line → glyph id sequence.
 * Sentences that yield < 2 tokens are skipped.
 *
 * Vocab is pre-seeded with the 88 canonical glyphs from the shared header —
 * no runtime vocab growth, so train and inference see the same token ids.
 */
static int load_stories(const char* path, GlyphVocab* vocab, StoryData* stories) {
    vocab->bos_id = semtok_seed_vocab(vocab->tokens, &vocab->vocab_size);

    FILE* f = fopen(path, "r");
    if (!f) { printf("Cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0) { fclose(f); return -1; }

    char* content = (char*)malloc((size_t)fsize + 1);
    if (!content) { fclose(f); return -1; }
    size_t got = fread(content, 1, (size_t)fsize, f);
    content[got] = '\0';
    fclose(f);

    stories->count = 0;
    long sent_start = 0;
    long total_sentences = 0, total_kept = 0;

    for (long i = 0; i <= fsize && stories->count < MAX_STORIES; i++) {
        int is_boundary = (i == fsize);
        if (!is_boundary && (content[i] == '.' || content[i] == '!' || content[i] == '?')) {
            char next = (i + 1 < fsize) ? content[i + 1] : ' ';
            if (next == ' ' || next == '\n' || next == '\r' || next == '"' ||
                next == '\'' || next == ')' || i + 1 >= fsize)
                is_boundary = 1;
        }
        if (!is_boundary) continue;

        long end = (i < fsize) ? i + 1 : i;
        long len = end - sent_start;
        sent_start = end;
        if (len < 3 || len >= 4096) continue;

        char sentence[4096];
        memcpy(sentence, content + (end - len), (size_t)len);
        sentence[len] = '\0';

        int has_alpha = 0;
        for (long j = 0; j < len; j++) {
            char c = sentence[j];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) { has_alpha = 1; break; }
        }
        if (!has_alpha) continue;
        total_sentences++;

        int toks[MAX_STORY_LEN];
        int n = semtok_line(sentence, toks, MAX_STORY_LEN - 2);
        if (n < 2) continue;

        int sid = stories->count;
        stories->lens[sid] = n;
        memcpy(stories->data[sid], toks, (size_t)n * sizeof(int));
        stories->count++;
        total_kept++;
    }

    free(content);
    printf("  source: %ld bytes, %ld sentences scanned, %ld kept (≥2 glyphs)\n",
           fsize, total_sentences, total_kept);
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
} GlyphModel;

static long count_params(GlyphModel* m, int V) {
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

static GlyphModel* model_create(int V) {
    /* V = vocab_size + 1 (BOS) */
    int Vp = V + 1;
    GlyphModel* m = (GlyphModel*)calloc(1, sizeof(GlyphModel));

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

static void model_free(GlyphModel* m) {
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

static int model_forward(GlyphModel* m, int* tokens, int* targets, int seq_len, int V) {
    int Vp = V + 1;

    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
    int wpe_i = nt_tape_param(m->wpe); nt_tape_no_decay(wpe_i);

    int li[8][8];
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

    int h = nt_seq_embedding(wte_i, wpe_i, tok_i, seq_len, E);

    for (int l = 0; l < N_L; l++) {
        int xn = nt_seq_rmsnorm(h, li[l][0], seq_len, E);
        int q = nt_seq_linear(li[l][1], xn, seq_len);
        int k = nt_seq_linear(li[l][2], xn, seq_len);
        int v = nt_seq_linear(li[l][3], xn, seq_len);
        int attn = nt_mh_causal_attention(q, k, v, seq_len, HD);
        int proj = nt_seq_linear(li[l][4], attn, seq_len);
        h = nt_add(h, proj);

        xn = nt_seq_rmsnorm(h, li[l][5], seq_len, E);
        int fc1 = nt_seq_linear(li[l][6], xn, seq_len);
        fc1 = nt_silu(fc1);
        int fc2 = nt_seq_linear(li[l][7], fc1, seq_len);
        h = nt_add(h, fc2);
    }

    int hf = nt_seq_rmsnorm(h, rmsf_i, seq_len, E);
    int logits = nt_seq_linear(head_i, hf, seq_len);

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
    {"small",    96, 8, 4, 128, 15000},
    {"medium",  128, 8, 4, 128, 15000},
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
    const char* dataset = "data/dracula.txt";
    const char* preset_name = "small";
    int    steps = 0;
    float  base_lr = 3e-4f;
    const char* save_path = "weights/cavellman_v3.bin";
    int    no_save = 0;
    int    seed = 42;

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
            printf("train_cavellman — caveLLMan training on notorch (pure C)\n\n");
            printf("  --dataset FILE    Raw English text (default: data/dracula.txt)\n");
            printf("  --preset NAME     tiny/micro/standard/small/medium (default: small)\n");
            printf("  --steps N         Training steps (default: preset)\n");
            printf("  --lr FLOAT        Learning rate (default: 3e-4)\n");
            printf("  --save FILE       Save weights (default: weights/cavellman_v3.bin)\n");
            printf("  --no-save         Don't save weights\n");
            printf("  --seed N          RNG seed (default: 42)\n");
            return 0;
        }
    }

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
    printf("  caveLLMan — Hieroglyphic LM on notorch (pure C)\n");
    printf("════════════════════════════════════════════════════════\n");
    printf("  dataset: %s\n", dataset);
    printf("  preset:  %s (E=%d H=%d L=%d CTX=%d)\n", preset_name, E, H, N_L, CTX);
    printf("  steps:   %d, lr=%.1e, seed=%d\n", steps, base_lr, seed);

    static GlyphVocab vocab;
    static StoryData stories;
    memset(&vocab, 0, sizeof(vocab));
    memset(&stories, 0, sizeof(stories));

    if (load_stories(dataset, &vocab, &stories) != 0) return 1;
    int V = vocab.vocab_size;
    printf("  stories: %d, vocab: %d glyphs + ⏹end = %d\n",
           stories.count, V, V + 1);

    printf("  vocab:   ");
    int show = V < 20 ? V : 20;
    for (int i = 0; i < show; i++) printf("%s ", vocab.tokens[i]);
    if (V > 20) printf("...");
    printf("\n");

    nt_seed(seed);
    GlyphModel* model = model_create(V);
    long np = count_params(model, V);
    printf("  params:  %ld (%.1f KB)\n", np, np * 4.0f / 1024.0f);
    printf("════════════════════════════════════════════════════════\n\n");

    nt_schedule sched = nt_schedule_cosine(base_lr, steps / 10, steps, base_lr * 0.1f);
    nt_nan_guard guard = nt_nan_guard_new();

    printf("training...\n");
    printf("──────────────────────────────────────────────────\n");

    double t0 = now_ms();
    float first_loss = 0, last_loss = 0;
    int log_every = steps > 200 ? steps / 50 : 4;
    if (log_every < 1) log_every = 1;

    for (int step = 0; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);

        int sid = step % stories.count;
        int slen = stories.lens[sid];

        int tokens[MAX_SEQ + 2], targets[MAX_SEQ + 2];
        int n = slen + 1;
        if (n > CTX) n = CTX;

        tokens[0] = vocab.bos_id;
        for (int i = 1; i < n; i++) {
            tokens[i] = (i - 1 < slen) ? stories.data[sid][i - 1] : vocab.bos_id;
        }
        for (int i = 0; i < n - 1; i++) {
            targets[i] = tokens[i + 1];
        }
        targets[n - 1] = vocab.bos_id;
        int seq = n;

        nt_tape_start();
        int loss_idx = model_forward(model, tokens, targets, seq, V);
        float loss_val = nt_tape_get()->entries[loss_idx].output->data[0];

        if (step == 0) first_loss = loss_val;
        last_loss = loss_val;

        nt_tape_backward(loss_idx);

        if (!nt_nan_guard_check(&guard)) {
            nt_tape_clear();
            continue;
        }

        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, loss_val);
        nt_tape_clear();

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
    printf("Training complete\n\n");

    /* ── Generation sample ────────────────────────────────────────────── */
    printf("── sample glyph sequences ──\n");
    nt_train_mode(0);

    int Vp = V + 1;
    for (int s = 0; s < 8; s++) {
        int ctx[MAX_SEQ];
        ctx[0] = vocab.bos_id;
        int gen_len = 1;

        for (int step = 0; step < CTX - 1 && gen_len < CTX; step++) {
            nt_tape_start();

            int toks[MAX_SEQ], tgts[MAX_SEQ];
            for (int i = 0; i < gen_len; i++) toks[i] = ctx[i];
            for (int i = gen_len; i < CTX; i++) toks[i] = 0;
            for (int i = 0; i < CTX; i++) tgts[i] = 0;

            int loss_idx_g = model_forward(model, toks, tgts, gen_len, V);

            nt_tape* tape = nt_tape_get();
            int logits_idx = tape->entries[loss_idx_g].parent1;
            nt_tensor* logits = tape->entries[logits_idx].output;

            float* last_logits = logits->data + (gen_len - 1) * Vp;
            float temp = 0.8f;

            float mx = last_logits[0];
            for (int i = 1; i < Vp; i++) if (last_logits[i] > mx) mx = last_logits[i];
            float sum = 0;
            float probs[MAX_VOCAB + 1];
            for (int i = 0; i < Vp; i++) {
                probs[i] = expf((last_logits[i] - mx) / temp);
                sum += probs[i];
            }
            for (int i = 0; i < Vp; i++) probs[i] /= sum;

            float top_p = 0.9f;
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
            float cum_p = 0;
            int nucleus_size = 0;
            for (int i = 0; i < Vp; i++) {
                cum_p += probs[indices[i]];
                nucleus_size++;
                if (cum_p >= top_p) break;
            }
            float nuc_sum = 0;
            for (int i = 0; i < nucleus_size; i++) nuc_sum += probs[indices[i]];
            float r = (float)rand() / (float)RAND_MAX;
            float cum = 0;
            int next = vocab.bos_id;
            for (int i = 0; i < nucleus_size; i++) {
                cum += probs[indices[i]] / nuc_sum;
                if (cum >= r) { next = indices[i]; break; }
            }

            nt_tape_clear();

            if (next == vocab.bos_id) break;
            ctx[gen_len++] = next;
        }

        printf("  %d: ", s + 1);
        for (int i = 1; i < gen_len; i++) {
            if (ctx[i] >= 0 && ctx[i] < V)
                printf("%s ", vocab.tokens[ctx[i]]);
        }
        printf("\n");
    }

    /* ── Save weights ──────────────────────────────────────────────────── */
    if (save_path && !no_save) {
        printf("\n── saving weights ──\n");
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
