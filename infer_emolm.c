/*
 * infer_emolm.c — Interactive emoji generation using trained emoLM weights
 *
 * Loads weights from nt_save binary + vocab file, then generates
 * emoji completions from user prompts interactively.
 *
 * Build: make infer
 * Run:   ./infer_emolm --weights weights/emolm.bin --preset tiny
 *        ./infer_emolm  (defaults: weights/emolm.bin, tiny preset)
 *
 * Copyright (C) 2026 Arianna Method contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ── Config ──────────────────────────────────────────────────────────────── */

#define MAX_VOCAB   256
#define MAX_SEQ     128

static int E     = 18;
static int H     = 3;
static int HD    = 6;
static int FFN_D = 72;
static int N_L   = 2;
static int CTX   = 32;

/* ── Model (same structure as train) ─────────────────────────────────────── */

typedef struct {
    nt_tensor *rms1, *wq, *wk, *wv, *wo;
    nt_tensor *rms2, *w_fc1, *w_fc2;
} Layer;

typedef struct {
    nt_tensor* wte;     /* [V+1, E] token embeddings */
    nt_tensor* wpe;     /* [CTX, E] position embeddings */
    Layer*     layers;
    nt_tensor* rms_f;   /* [E] final rmsnorm */
    nt_tensor* head;    /* [V+1, E] output projection */
} EmoModel;

/* ── Presets ──────────────────────────────────────────────────────────────── */

typedef struct { const char* name; int embd, heads, layers, ctx; } Preset;

static const Preset PRESETS[] = {
    {"tiny",     18, 3, 2, 32},
    {"micro",    48, 4, 3, 64},
    {"standard", 64, 8, 3, 64},
    {"small",    96, 8, 4, 128},
    {"medium",  128, 8, 4, 128},
};
#define N_PRESETS 5

static const Preset* find_preset(const char* name) {
    for (int i = 0; i < N_PRESETS; i++)
        if (strcmp(PRESETS[i].name, name) == 0) return &PRESETS[i];
    return NULL;
}

/* ── Load vocab ──────────────────────────────────────────────────────────── */

typedef struct {
    char tokens[MAX_VOCAB][32];
    int  vocab_size;
    int  bos_id;
} EmojiVocab;

static int load_vocab(const char* path, EmojiVocab* v) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("Error: cannot open vocab %s\n", path); return -1; }

    char line[256];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    v->vocab_size = atoi(line);
    if (v->vocab_size <= 0 || v->vocab_size >= MAX_VOCAB) {
        printf("Error: invalid vocab size %d\n", v->vocab_size);
        fclose(f); return -1;
    }

    for (int i = 0; i < v->vocab_size; i++) {
        if (!fgets(line, sizeof(line), f)) break;
        /* Strip newline */
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        strncpy(v->tokens[i], line, 31);
        v->tokens[i][31] = '\0';
    }
    v->bos_id = v->vocab_size;
    fclose(f);
    return 0;
}

static int vocab_find(EmojiVocab* v, const char* tok) {
    for (int i = 0; i < v->vocab_size; i++)
        if (strcmp(v->tokens[i], tok) == 0) return i;
    return -1;
}

/* ── Model forward (inference only) ──────────────────────────────────────── */

static int model_forward(EmoModel* model, int* tokens, int* targets, int seq_len, int V) {
    int Vp = V + 1;

    int wte_i = nt_tape_param(model->wte);
    int wpe_i = nt_tape_param(model->wpe);

    /* Layer indices */
    int li[16][8];
    for (int l = 0; l < N_L; l++) {
        li[l][0] = nt_tape_param(model->layers[l].rms1);
        li[l][1] = nt_tape_param(model->layers[l].wq);
        li[l][2] = nt_tape_param(model->layers[l].wk);
        li[l][3] = nt_tape_param(model->layers[l].wv);
        li[l][4] = nt_tape_param(model->layers[l].wo);
        li[l][5] = nt_tape_param(model->layers[l].rms2);
        li[l][6] = nt_tape_param(model->layers[l].w_fc1);
        li[l][7] = nt_tape_param(model->layers[l].w_fc2);
    }
    int rmsf_i = nt_tape_param(model->rms_f);
    int head_i = nt_tape_param(model->head);

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

/* ── Rebuild model structure and load weights ────────────────────────────── */

static EmoModel* model_load(const char* weights_path, int V) {
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(weights_path, &n_loaded);
    if (!loaded || n_loaded == 0) {
        printf("Error: failed to load weights from %s\n", weights_path);
        return NULL;
    }

    int expected = 2 + N_L * 8 + 2; /* wte, wpe, 8 per layer, rms_f, head */
    if (n_loaded < expected) {
        printf("Error: weight file has %d tensors, need %d for %d layers\n",
               n_loaded, expected, N_L);
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded);
        return NULL;
    }

    EmoModel* m = (EmoModel*)calloc(1, sizeof(EmoModel));
    m->layers = (Layer*)calloc(N_L, sizeof(Layer));

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
    }
    m->rms_f = loaded[pi++];
    m->head  = loaded[pi++];

    printf("  loaded %d/%d tensors\n", pi, n_loaded);
    free(loaded);
    return m;
}

/* ── Top-p nucleus sampling ──────────────────────────────────────────────── */

static int sample_top_p(float* logits, int n, float temp, float top_p) {
    /* Temperature + softmax */
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];

    float probs[MAX_VOCAB + 1];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((logits[i] - mx) / temp);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;

    /* Sort indices by prob descending */
    int indices[MAX_VOCAB + 1];
    for (int i = 0; i < n; i++) indices[i] = i;
    for (int i = 1; i < n; i++) {
        int key = indices[i];
        float kp = probs[key];
        int j = i - 1;
        while (j >= 0 && probs[indices[j]] < kp) {
            indices[j + 1] = indices[j]; j--;
        }
        indices[j + 1] = key;
    }

    /* Accumulate nucleus */
    float cum_p = 0;
    int nuc_sz = 0;
    for (int i = 0; i < n; i++) {
        cum_p += probs[indices[i]];
        nuc_sz++;
        if (cum_p >= top_p) break;
    }

    /* Renormalize + sample */
    float nuc_sum = 0;
    for (int i = 0; i < nuc_sz; i++) nuc_sum += probs[indices[i]];

    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0;
    for (int i = 0; i < nuc_sz; i++) {
        cum += probs[indices[i]] / nuc_sum;
        if (cum >= r) return indices[i];
    }
    return indices[0];
}

/* ── Tokenize input prompt ───────────────────────────────────────────────── */

static int tokenize_prompt(const char* input, EmojiVocab* vocab, int* tokens, int max_tokens) {
    int n = 0;
    char buf[256];
    strncpy(buf, input, 255); buf[255] = '\0';

    char* tok = strtok(buf, " \t\n");
    while (tok && n < max_tokens) {
        int id = vocab_find(vocab, tok);
        if (id >= 0) {
            tokens[n++] = id;
        }
        tok = strtok(NULL, " \t\n");
    }
    return n;
}

/* ── Generate from prompt ────────────────────────────────────────────────── */

static void generate(EmoModel* model, EmojiVocab* vocab, int* prompt, int prompt_len,
                     int max_gen, float temp, float top_p) {
    int V = vocab->vocab_size;
    int Vp = V + 1;
    int ctx[MAX_SEQ];
    int gen_len = 0;

    /* BOS + prompt */
    ctx[gen_len++] = vocab->bos_id;
    for (int i = 0; i < prompt_len && gen_len < CTX; i++)
        ctx[gen_len++] = prompt[i];

    /* Print prompt */
    for (int i = 0; i < prompt_len; i++)
        printf("%s ", vocab->tokens[prompt[i]]);

    nt_train_mode(0);

    for (int step = 0; step < max_gen && gen_len < CTX; step++) {
        nt_tape_start();

        int toks[MAX_SEQ], tgts[MAX_SEQ];
        for (int i = 0; i < gen_len; i++) toks[i] = ctx[i];
        for (int i = gen_len; i < CTX; i++) toks[i] = 0;
        for (int i = 0; i < CTX; i++) tgts[i] = 0;

        int loss_idx = model_forward(model, toks, tgts, gen_len, V);

        /* Get logits */
        nt_tape* tape = nt_tape_get();
        int logits_idx = tape->entries[loss_idx].parent1;
        nt_tensor* logits = tape->entries[logits_idx].output;
        float* last_logits = logits->data + (gen_len - 1) * Vp;

        int next = sample_top_p(last_logits, Vp, temp, top_p);
        nt_tape_clear();

        if (next == vocab->bos_id) break;

        ctx[gen_len++] = next;
        if (next >= 0 && next < V)
            printf("%s ", vocab->tokens[next]);
        fflush(stdout);
    }
    printf("\n");
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    const char* weights_path = "weights/emolm.bin";
    const char* preset_name = "tiny";
    float temp = 0.8f;
    float top_p = 0.9f;
    int max_gen = 0; /* 0 = CTX - prompt_len */
    int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc)
            weights_path = argv[++i];
        else if (strcmp(argv[i], "--preset") == 0 && i + 1 < argc)
            preset_name = argv[++i];
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc)
            temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc)
            top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--max") == 0 && i + 1 < argc)
            max_gen = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("infer_emolm — interactive emoji generation (pure C)\n\n");
            printf("  --weights FILE  Weight file (default: weights/emolm.bin)\n");
            printf("  --preset NAME   Model preset (default: tiny)\n");
            printf("  --temp FLOAT    Temperature (default: 0.8)\n");
            printf("  --top-p FLOAT   Nucleus sampling threshold (default: 0.9)\n");
            printf("  --max N         Max tokens to generate (default: CTX)\n");
            printf("  --seed N        RNG seed (default: 42)\n");
            printf("\nInteractive mode: type emoji prompt, get completion.\n");
            printf("Type 'quit' or Ctrl-D to exit.\n");
            return 0;
        }
    }

    /* Apply preset */
    const Preset* pr = find_preset(preset_name);
    if (!pr) {
        printf("Unknown preset: %s\n", preset_name);
        return 1;
    }
    E = pr->embd; H = pr->heads; HD = E / H;
    FFN_D = 4 * E; N_L = pr->layers; CTX = pr->ctx;
    if (max_gen <= 0) max_gen = CTX - 1;

    srand(seed);

    printf("════════════════════════════════════════════════════════\n");
    printf("  emoLM — Interactive Emoji Generation (notorch C)\n");
    printf("════════════════════════════════════════════════════════\n");
    printf("  weights: %s\n", weights_path);
    printf("  preset:  %s (E=%d H=%d L=%d CTX=%d)\n", preset_name, E, H, N_L, CTX);
    printf("  temp=%.2f, top_p=%.2f, max=%d\n", temp, top_p, max_gen);

    /* Load vocab */
    char vocab_path[512];
    snprintf(vocab_path, sizeof(vocab_path), "%s.vocab", weights_path);
    EmojiVocab vocab;
    memset(&vocab, 0, sizeof(vocab));
    if (load_vocab(vocab_path, &vocab) != 0) return 1;
    printf("  vocab: %d emojis\n", vocab.vocab_size);

    /* Load model */
    printf("  loading weights...\n");
    nt_seed(seed);
    EmoModel* model = model_load(weights_path, vocab.vocab_size);
    if (!model) return 1;
    printf("════════════════════════════════════════════════════════\n\n");

    /* Interactive loop */
    printf("Enter emoji prompt (space-separated), or 'quit' to exit:\n\n");
    char input[1024];
    while (1) {
        printf("🎯 > ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;

        /* Strip newline */
        int len = (int)strlen(input);
        while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r')) input[--len] = '\0';
        if (len == 0) continue;
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) break;

        /* Tokenize prompt */
        int prompt_tokens[MAX_SEQ];
        int n_prompt = tokenize_prompt(input, &vocab, prompt_tokens, CTX - 2);

        if (n_prompt == 0) {
            printf("  (no recognized emojis in prompt, generating from scratch)\n");
        }

        printf("  → ");
        generate(model, &vocab, prompt_tokens, n_prompt, max_gen, temp, top_p);
        printf("\n");
    }

    printf("\nbye 👋\n");
    /* model tensors were allocated by nt_load, no custom free needed */
    free(model->layers);
    free(model);
    return 0;
}
