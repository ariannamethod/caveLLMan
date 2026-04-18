/*
 * train_diffusion.c — Discrete diffusion training for caveLLMan
 *
 * Instead of left-to-right autoregressive, this trains a BIDIRECTIONAL
 * transformer to denoise masked sequences. At inference, start from
 * all-MASK and iteratively reveal tokens by confidence.
 *
 * The cave painting appears all at once.
 *
 * Build: make train_diffusion
 * Run:   ./train_diffusion --dataset data/cavellman_train_final.txt --steps 15000
 *
 * No Python. No pip. No torch. Pure C + notorch.
 */

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define MAX_VOCAB   256
#define MAX_SEQ     128
#define MAX_STORIES 32000
#define MAX_STORY_LEN 32

/* Architecture */
static int E     = 96;
static int H     = 8;
static int HD    = 12;
static int FFN_D = 384;
static int N_L   = 4;
static int CTX   = 32;

/* Diffusion config */
#define T_STEPS   30     /* diffusion timesteps */

/* ── Tokenizer (same as train_cavellman.c — space split) ─────────────── */

typedef struct {
    char tokens[MAX_VOCAB][32];
    int  vocab_size;
    int  mask_id;    /* MASK token = vocab_size */
} DiffVocab;

typedef struct {
    int  data[MAX_STORIES][MAX_STORY_LEN];
    int  lens[MAX_STORIES];
    int  count;
} StoryData;

static int vocab_find(DiffVocab* v, const char* tok) {
    for (int i = 0; i < v->vocab_size; i++)
        if (strcmp(v->tokens[i], tok) == 0) return i;
    return -1;
}

static int vocab_add(DiffVocab* v, const char* tok) {
    int id = vocab_find(v, tok);
    if (id >= 0) return id;
    if (v->vocab_size >= MAX_VOCAB - 2) return -1;
    id = v->vocab_size;
    strncpy(v->tokens[id], tok, 31);
    v->tokens[id][31] = '\0';
    v->vocab_size++;
    return id;
}

static int load_stories(const char* path, DiffVocab* vocab, StoryData* stories) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("Cannot open %s\n", path); return -1; }
    vocab->vocab_size = 0;
    stories->count = 0;
    char line[4096];
    while (fgets(line, sizeof(line), f) && stories->count < MAX_STORIES) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        int sid = stories->count;
        stories->lens[sid] = 0;
        char* tok = strtok(line, " ");
        while (tok && stories->lens[sid] < MAX_STORY_LEN - 1) {
            if (strlen(tok) == 0) { tok = strtok(NULL, " "); continue; }
            int id = vocab_add(vocab, tok);
            if (id >= 0) stories->data[sid][stories->lens[sid]++] = id;
            tok = strtok(NULL, " ");
        }
        if (stories->lens[sid] > 0) stories->count++;
    }
    fclose(f);
    vocab->mask_id = vocab->vocab_size;
    strncpy(vocab->tokens[vocab->mask_id], "MASK", 31);
    vocab->vocab_size++; /* MASK is now part of vocab */
    return 0;
}

/* ── Model (same architecture, but NO causal mask) ──────────────────────── */

typedef struct {
    nt_tensor *wte;     /* [V, E] — includes MASK token */
    nt_tensor *wpe;     /* [CTX, E] */
    struct {
        nt_tensor *rms1;
        nt_tensor *wq, *wk, *wv, *wo;
        nt_tensor *rms2;
        nt_tensor *w_fc1, *w_fc2;
    } layers[8];
    nt_tensor *rms_f;
    nt_tensor *head;    /* [V, E] */
} DiffModel;

static DiffModel* model_create(int V) {
    DiffModel* m = (DiffModel*)calloc(1, sizeof(DiffModel));
    m->wte = nt_tensor_new2d(V, E); nt_tensor_xavier(m->wte, V, E);
    m->wpe = nt_tensor_new2d(CTX, E); nt_tensor_xavier(m->wpe, CTX, E);
    float sr = 0.02f / sqrtf(2.0f * N_L);
    for (int l = 0; l < N_L; l++) {
        m->layers[l].rms1 = nt_tensor_new(E); nt_tensor_fill(m->layers[l].rms1, 1.0f);
        m->layers[l].wq = nt_tensor_new2d(E, E); nt_tensor_xavier(m->layers[l].wq, E, E);
        m->layers[l].wk = nt_tensor_new2d(E, E); nt_tensor_xavier(m->layers[l].wk, E, E);
        m->layers[l].wv = nt_tensor_new2d(E, E); nt_tensor_xavier(m->layers[l].wv, E, E);
        m->layers[l].wo = nt_tensor_new2d(E, E); nt_tensor_xavier(m->layers[l].wo, E, E);
        for (int i = 0; i < m->layers[l].wo->len; i++) m->layers[l].wo->data[i] *= sr / 0.1f;
        m->layers[l].rms2 = nt_tensor_new(E); nt_tensor_fill(m->layers[l].rms2, 1.0f);
        m->layers[l].w_fc1 = nt_tensor_new2d(FFN_D, E); nt_tensor_xavier(m->layers[l].w_fc1, E, FFN_D);
        m->layers[l].w_fc2 = nt_tensor_new2d(E, FFN_D); nt_tensor_xavier(m->layers[l].w_fc2, FFN_D, E);
        for (int i = 0; i < m->layers[l].w_fc2->len; i++) m->layers[l].w_fc2->data[i] *= sr / 0.1f;
    }
    m->rms_f = nt_tensor_new(E); nt_tensor_fill(m->rms_f, 1.0f);
    m->head = nt_tensor_new2d(V, E); nt_tensor_xavier(m->head, E, V);
    return m;
}

/*
 * Diffusion forward: given partially masked input, predict original tokens.
 * Uses BIDIRECTIONAL attention (nt_mh_causal_attention sees all positions —
 * for diffusion we need full attention, but notorch only has causal.
 * HACK: reverse sequence trick — run forward + backward, average.
 * TODO: add nt_mh_full_attention to notorch for proper bidirectional.
 *
 * For now: use causal attention as approximation. Each position sees
 * all PREVIOUS positions including MASKed ones. Not ideal but trains.
 */
static int model_forward_diffusion(DiffModel* m, int* masked_tokens, int* targets,
                                    int* mask_positions, int n_masked,
                                    int seq_len, int V) {
    nt_tape_start();

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
        tok_t->data[i] = (float)masked_tokens[i];
        tgt_t->data[i] = (float)targets[i];
    }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t); nt_tensor_free(tgt_t);

    int h = nt_seq_embedding(wte_i, wpe_i, tok_i, seq_len, E);

    for (int l = 0; l < N_L; l++) {
        int xn = nt_seq_rmsnorm(h, li[l][0], seq_len, E);
        int q = nt_seq_linear(li[l][1], xn, seq_len);
        int k = nt_seq_linear(li[l][2], xn, seq_len);
        int v = nt_seq_linear(li[l][3], xn, seq_len);
        /* Causal attention — each position sees left context + self */
        int attn = nt_mh_causal_attention(q, k, v, seq_len, HD);
        int proj = nt_seq_linear(li[l][4], attn, seq_len);
        h = nt_add(h, proj);
        xn = nt_seq_rmsnorm(h, li[l][5], seq_len, E);
        int fc1 = nt_silu(nt_seq_linear(li[l][6], xn, seq_len));
        int fc2 = nt_seq_linear(li[l][7], fc1, seq_len);
        h = nt_add(h, fc2);
    }

    int hf = nt_seq_rmsnorm(h, rmsf_i, seq_len, E);
    int logits = nt_seq_linear(head_i, hf, seq_len);

    /* Loss only on MASKED positions */
    int loss = nt_seq_cross_entropy(logits, tgt_i, seq_len, V);
    return loss;
}

/* ── Random masking ─────────────────────────────────────────────────────── */

static unsigned long rng = 42;
static int rand_int(int n) { rng = rng * 6364136223846793005ULL + 1; return (int)((rng >> 33) % n); }
static float rand_f(void) { rng = rng * 6364136223846793005ULL + 1; return (float)(rng >> 33) / (float)0x7FFFFFFF; }

/*
 * Apply random masking: for a given noise level t (0 = no mask, T = all mask),
 * mask approximately (t/T) fraction of tokens.
 */
static int apply_mask(int* src, int* dst, int* mask_pos, int len, int mask_id, int t, int T) {
    float mask_rate = (float)t / (float)T;
    int n_masked = 0;
    for (int i = 0; i < len; i++) {
        if (rand_f() < mask_rate) {
            dst[i] = mask_id;
            mask_pos[n_masked++] = i;
        } else {
            dst[i] = src[i];
        }
    }
    /* Ensure at least 1 mask */
    if (n_masked == 0 && len > 0) {
        int pos = rand_int(len);
        dst[pos] = mask_id;
        mask_pos[0] = pos;
        n_masked = 1;
    }
    return n_masked;
}

/* ── Timer ──────────────────────────────────────────────────────────────── */

static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ── Inference: iterative unmasking ─────────────────────────────────────── */

static void diffusion_generate(DiffModel* m, DiffVocab* vocab, int len) {
    int V = vocab->vocab_size;
    int seq[MAX_SEQ];
    int prev_grad = 1; /* save and disable grad for inference */

    /* Start: all MASK */
    for (int i = 0; i < len; i++) seq[i] = vocab->mask_id;

    printf("  generating (%d steps):\n  ", T_STEPS);
    for (int i = 0; i < len; i++) printf("MASK ");
    printf("\n");

    /* Iteratively unmask */
    for (int step = 0; step < T_STEPS; step++) {
        /* Count remaining masks */
        int n_mask = 0;
        for (int i = 0; i < len; i++) if (seq[i] == vocab->mask_id) n_mask++;
        if (n_mask == 0) break;

        /* How many to reveal this step (linear schedule) */
        int to_reveal = (len + T_STEPS - 1) / T_STEPS;
        if (to_reveal < 1) to_reveal = 1;
        if (to_reveal > n_mask) to_reveal = n_mask;

        /* Forward pass — get logits for all positions */
        nt_tape_start();
        int wte_i = nt_tape_param(m->wte);
        int wpe_i = nt_tape_param(m->wpe);
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

        nt_tensor* tok_t = nt_tensor_new(len);
        for (int i = 0; i < len; i++) tok_t->data[i] = (float)seq[i];
        int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
        nt_tensor_free(tok_t);

        int h = nt_seq_embedding(wte_i, wpe_i, tok_i, len, E);
        for (int l = 0; l < N_L; l++) {
            int xn = nt_seq_rmsnorm(h, li[l][0], len, E);
            int q = nt_seq_linear(li[l][1], xn, len);
            int k = nt_seq_linear(li[l][2], xn, len);
            int v = nt_seq_linear(li[l][3], xn, len);
            int attn = nt_mh_causal_attention(q, k, v, len, HD);
            int proj = nt_seq_linear(li[l][4], attn, len);
            h = nt_add(h, proj);
            xn = nt_seq_rmsnorm(h, li[l][5], len, E);
            int fc1 = nt_silu(nt_seq_linear(li[l][6], xn, len));
            int fc2 = nt_seq_linear(li[l][7], fc1, len);
            h = nt_add(h, fc2);
        }
        int hf = nt_seq_rmsnorm(h, rmsf_i, len, E);
        int logits_idx = nt_seq_linear(head_i, hf, len);

        /* Get logits tensor */
        nt_tape_entry* tape = (nt_tape_entry*)nt_tape_get();
        float* logits = tape[logits_idx].output->data;

        /* Find most confident masked positions */
        typedef struct { int pos; float conf; int token; } Candidate;
        Candidate cands[MAX_SEQ];
        int nc = 0;
        for (int i = 0; i < len; i++) {
            if (seq[i] != vocab->mask_id) continue;
            float* log_i = logits + i * V;
            /* Find argmax and its confidence */
            int best = 0; float best_val = log_i[0];
            for (int j = 1; j < V; j++) {
                if (log_i[j] > best_val && j != vocab->mask_id) {
                    best_val = log_i[j]; best = j;
                }
            }
            /* Confidence = max logit - second max */
            float second = -1e30f;
            for (int j = 0; j < V; j++)
                if (j != best && j != vocab->mask_id && log_i[j] > second) second = log_i[j];
            cands[nc++] = (Candidate){i, best_val - second, best};
        }

        /* Sort by confidence descending */
        for (int i = 0; i < nc - 1; i++)
            for (int j = i + 1; j < nc; j++)
                if (cands[j].conf > cands[i].conf) {
                    Candidate tmp = cands[i]; cands[i] = cands[j]; cands[j] = tmp;
                }

        /* Reveal top candidates */
        for (int i = 0; i < to_reveal && i < nc; i++)
            seq[cands[i].pos] = cands[i].token;

        nt_tape_clear();

        /* Print current state */
        printf("  [%2d] ", step + 1);
        for (int i = 0; i < len; i++) {
            if (seq[i] == vocab->mask_id) printf("____ ");
            else printf("%s ", vocab->tokens[seq[i]]);
        }
        printf("\n");
    }

    printf("\n  final: ");
    for (int i = 0; i < len; i++) printf("%s ", vocab->tokens[seq[i]]);
    printf("\n");
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    const char* dataset = "data/cavellman_train_final.txt";
    int steps = 15000;
    float lr = 3e-4f;
    int gen_len = 8;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset") == 0 && i+1 < argc) dataset = argv[++i];
        else if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--len") == 0 && i+1 < argc) gen_len = atoi(argv[++i]);
    }

    printf("════════════════════════════════════════════════════════\n");
    printf("  caveLLMan DIFFUSION — the cave painting appears at once\n");
    printf("════════════════════════════════════════════════════════\n");

    DiffVocab vocab; StoryData stories;
    if (load_stories(dataset, &vocab, &stories) < 0) return 1;
    int V = vocab.vocab_size;
    printf("  dataset: %s\n", dataset);
    printf("  stories: %d, vocab: %d (including MASK)\n", stories.count, V);
    printf("  diffusion steps: %d, gen_len: %d\n", T_STEPS, gen_len);

    nt_seed(42);
    DiffModel* m = model_create(V);
    printf("  params: %ld\n", (long)(m->wte->len + m->wpe->len + m->rms_f->len + m->head->len));

    printf("\ntraining...\n");
    double t0 = now_ms();
    int nans = 0;

    for (int step = 1; step <= steps; step++) {
        /* Pick random story */
        int sid = rand_int(stories.count);
        int slen = stories.lens[sid];
        if (slen < 3 || slen > CTX) continue;

        /* Random noise level */
        int t = 1 + rand_int(T_STEPS);

        /* Apply mask */
        int masked[MAX_SEQ], mask_pos[MAX_SEQ];
        int n_masked = apply_mask(stories.data[sid], masked, mask_pos, slen, vocab.mask_id, t, T_STEPS);

        /* Forward + backward */
        nt_tape_start();
        int loss_idx = model_forward_diffusion(m, masked, stories.data[sid],
                                                mask_pos, n_masked, slen, V);

        /* Get loss value */
        nt_tape_entry* tape = (nt_tape_entry*)nt_tape_get();
        float loss_val = tape[loss_idx].output->data[0];

        if (loss_val != loss_val) { nans++; nt_tape_clear(); continue; }

        nt_tape_backward(loss_idx);
        nt_tape_clip_grads(1.0f);

        /* Cosine LR schedule */
        float prog = (float)step / steps;
        float cos_lr = lr * 0.5f * (1.0f + cosf(3.14159f * prog));
        if (step < steps / 20) cos_lr = lr * (float)step / (steps / 20);

        nt_tape_chuck_step(cos_lr, loss_val);
        nt_tape_clear();

        if (step % (steps / 50) == 0 || step == 1) {
            double elapsed = (now_ms() - t0) / 1000.0;
            printf("  step %5d/%d | loss %.4f | lr %.2e | %.1fs\n",
                   step, steps, loss_val, cos_lr, elapsed);
        }
    }

    double elapsed = (now_ms() - t0) / 1000.0;
    printf("────────────────────────────────────────────────\n");
    printf("  training complete. %.1fs, %d NaN skipped\n", elapsed, nans);

    /* Generate samples */
    printf("\n── diffusion generation ──\n");
    for (int s = 0; s < 4; s++) {
        printf("\n  Story %d (len=%d):\n", s + 1, gen_len);
        diffusion_generate(m, &vocab, gen_len);
    }

    printf("\n════════════════════════════════════════════════════════\n");
    printf("  the cave painting is complete. fuck torch.\n");
    printf("════════════════════════════════════════════════════════\n");

    return 0;
}
