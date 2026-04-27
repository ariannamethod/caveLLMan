/*
 * trinity.c — Molly + cosmic-physics affair mitosis + jealousy field.
 *
 * Extracted from cavellman_42.c into its own module (Олег 2026-04-27:
 * "разделить на модули, начиная с какой-то версии, например 42").
 * Predator already extracted in predator.c; this is the second module.
 *
 * Contents:
 *   - g_molly + g_molly_lock (state)
 *   - cosmic_tension(t) / ring_coherence_metric() — AML field functions
 *   - find_family_pair / find_affair_mate — mitosis path selectors
 *   - colony_affair_with_molly(mate, preset) — affair child creation
 *   - colony_try_mitosis_trinity(preset) — trinity-mode top-level mitosis
 *   - molly_thread_main(arg) — Molly's horizon-goroutine loop
 *   - blend_affair_surface(mate, child) — cooccur-only memetic blend
 *
 * Cross-module surface (declared in cavellman_42.h):
 *   - extern MollyState g_molly, extern pthread_mutex_t g_molly_lock
 *   - colony_try_mitosis_trinity(), molly_thread_main()
 *
 * Static helpers (file-scope here): cosmic_tension, ring_coherence_metric,
 * find_family_pair, find_affair_mate, colony_affair_with_molly,
 * blend_affair_surface.
 */
#include "cavellman_42.h"
#include "semantic_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>

#define LOVER_BASELINE        0.20f
#define MITOSIS_MIN_TURNS     30
#define JEALOUSY_DISSONANCE   0.30f
#define JEALOUSY_FLOOR_BUMP   0.05f
#define MOLLY_DNA_DIR         "dna/output/molly"
#define MOLLY_TICK_USEC       300000   /* 0.3s — slower than cave threads */
#define DUAL_MAX_GEN_LOC      24

/* Cross-module helpers also declared in cavellman_42.h. */
/* All cross-module externs declared in cavellman_42.h. */

/* Forward decl — defined later in this file. */
static int blend_affair_surface(const Cave* mate, Cave* child);

MollyState g_molly = {0};
pthread_mutex_t g_molly_lock = PTHREAD_MUTEX_INITIALIZER;

/* ── Cosmic-physics affair driver ──────────────────────────────────────
 * affair_prob = clip(0.2 + cosmic_tension - 0.5·ring_coherence, 0, 1)
 * cosmic_tension(t) = (sin(t/86400 * 2π) + 1)/2  — 24h sinusoid.
 * High-tension + low-coherence days = affair days. Calm days = family. */
static float cosmic_tension(time_t now) {
    float day = (float)(now % 86400) / 86400.0f;
    float v = (sinf(day * 6.28318530f) + 1.0f) * 0.5f;
    return v;
}

static float ring_coherence_metric(void) {
    if (g_colony_n < 1) return 1.0f;
    float sum = 0.0f;
    for (int ci = 0; ci < g_colony_n; ci++)
        sum += g_colony[ci]->field.dissonance;
    float mean_diss = sum / (float)g_colony_n;
    return 1.0f - mean_diss;
}

static int find_family_pair(int* a_out, int* b_out) {
    int best_a = -1, best_b = -1;
    float best_score = -1.0f;
    for (int i = 0; i < g_colony_n; i++) {
        if (g_colony[i]->is_lover) continue;
        for (int j = i + 1; j < g_colony_n; j++) {
            if (g_colony[j]->is_lover) continue;
            float s = g_colony[i]->field.total_count +
                      g_colony[j]->field.total_count;
            if (g_colony[i]->field.total_count < MITOSIS_MIN_TURNS ||
                g_colony[j]->field.total_count < MITOSIS_MIN_TURNS) continue;
            if (s > best_score) { best_score = s; best_a = i; best_b = j; }
        }
    }
    if (best_a < 0) return 0;
    *a_out = best_a; *b_out = best_b;
    return 1;
}

static int find_affair_mate(void) {
    int best = -1;
    float best_score = -1.0f;
    for (int i = 0; i < g_colony_n; i++) {
        if (g_colony[i]->is_lover) continue;
        if (g_colony[i]->field.total_count < MITOSIS_MIN_TURNS) continue;
        float s = g_colony[i]->field.excitement +
                  0.5f * g_colony[i]->field.dissonance;
        if (s > best_score) { best_score = s; best = i; }
    }
    return best;
}

/* Cave × Molly affair → bastard child. Memetic substrate via
 * blend_affair_surface (cooccur-only, conservative). */
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

    if (blend_weights(pa->field.weights_path, g_molly.weights_path, child_w) != 0) {
        printf("  [affair] blend_weights failed for %s × Molly → %s\n",
               pa->field.name, child_w);
        return NULL;
    }
    copy_file(parent_v, child_v);
    copy_file(parent_j, child_j);

    float child_baseline = 0.5f * (pa->field.baseline_floor + LOVER_BASELINE);
    char* permanent_name = strdup(child_name);
    Cave* child = cave_new(permanent_name, child_baseline, child_w, preset_name);
    if (!child) { free(permanent_name); return NULL; }

    blend_affair_surface(pa, child);

    {
        const int* src_tok = pa->last_tokens;
        int src_len = pa->last_len;
        pthread_mutex_lock(&g_molly_lock);
        if ((rand() % 2) && g_molly.last_len > 0) {
            src_tok = g_molly.last_tokens;
            src_len = g_molly.last_len;
        }
        if (src_len > 32) src_len = 32;
        if (src_len > 0)
            memcpy(child->last_tokens, src_tok, (size_t)src_len * sizeof(int));
        child->last_len = src_len;
        pthread_mutex_unlock(&g_molly_lock);
    }
    child->spore_saved_at = (int64_t)time(NULL);
    child->field.immunity_ticks = NEWBORN_IMMUNITY_TICKS;
    child->is_bastard = 1;

    colony_add(child);
    g_children_born++;
    return child;
}

/* Trinity-mode top-level mitosis: family vs affair chosen by AML cosmic
 * physics, jealousy spike on non-parent caves after affair child birth. */
void colony_try_mitosis_trinity(const char* preset_name) {
    if (g_mitosis_cooldown > 0) { g_mitosis_cooldown--; return; }
    if (g_colony_n < 2) return;

    float ct = cosmic_tension(time(NULL));
    float coh = ring_coherence_metric();
    float affair_prob = 0.2f + ct - 0.5f * coh;
    if (affair_prob < 0.0f) affair_prob = 0.0f;
    if (affair_prob > 1.0f) affair_prob = 1.0f;

    float roll = (float)rand() / (float)RAND_MAX;
    int do_affair = (roll < affair_prob) && (g_molly.model != NULL);

    Cave* child = NULL;
    int pa_idx = -1, pb_idx = -1;

    if (do_affair) {
        pa_idx = find_affair_mate();
        if (pa_idx < 0) {
            do_affair = 0;
        } else {
            child = colony_affair_with_molly(pa_idx, preset_name);
            if (!child) return;
            printf("\n  *** AFFAIR MITOSIS: M × %s → %s "
                   "(cosmic %.2f coh %.2f prob %.2f) ***\n",
                   g_colony[pa_idx]->field.name, child->field.name,
                   ct, coh, affair_prob);
        }
    }

    if (!do_affair) {
        if (!find_family_pair(&pa_idx, &pb_idx)) return;
        printf("\n  *** FAMILY MITOSIS: %s × %s "
               "(cosmic %.2f coh %.2f prob %.2f sober) ***\n",
               g_colony[pa_idx]->field.name, g_colony[pb_idx]->field.name,
               ct, coh, affair_prob);
        /* Family path delegates to existing colony_mitosis (defined in
         * cavellman_42.c). */
        extern Cave* colony_mitosis(const Cave* pa, const Cave* pb,
                                     const char* preset_name);
        child = colony_mitosis(g_colony[pa_idx], g_colony[pb_idx], preset_name);
        if (!child) return;
    }

    /* Jealousy field event after affair birth — non-parent caves spike. */
    if (do_affair) {
        int n_jealous = 0;
        for (int ci = 0; ci < g_colony_n; ci++) {
            Cave* c = g_colony[ci];
            if (ci == pa_idx) continue;
            if (c == child) continue;
            c->field.dissonance += JEALOUSY_DISSONANCE;
            if (c->field.dissonance > 1.0f) c->field.dissonance = 1.0f;
            float ceil = c->field.baseline_floor + MATURITY_CAP;
            float new_floor = c->field.coherence_floor + JEALOUSY_FLOOR_BUMP;
            if (new_floor > ceil) new_floor = ceil;
            c->field.coherence_floor = new_floor;
            n_jealous++;
        }
        if (n_jealous > 0)
            printf("  [jealousy] %d cave(s): dissonance +%.2f, floor +%.2f\n",
                   n_jealous, JEALOUSY_DISSONANCE, JEALOUSY_FLOOR_BUMP);
    }

    g_mitosis_cooldown = MITOSIS_COOLDOWN_TICKS;
}

/* Molly's horizon-goroutine: continuously generates utterances and
 * writes them to dna/output/molly/ where the existing learner picks
 * them up. Holds g_learner.lock during dual_generate (KV cache is
 * global static, shared with cave threads). */

void* molly_thread_main(void* arg) {
    (void)arg;
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

        pthread_mutex_lock(&g_learner.lock);
        pthread_mutex_lock(&g_molly_lock);
        int gen[MAX_SEQ];
        int gn = dual_generate(g_molly.model, g_molly.vocab,
                               prompt, prompt_len,
                               DUAL_MAX_GEN_LOC, 0.8f, 0.9f, gen, MAX_SEQ);
        pthread_mutex_unlock(&g_molly_lock);
        pthread_mutex_unlock(&g_learner.lock);

        if (gn > 0) {
            printf("[M] ");
            for (int i = 0; i < gn; i++) {
                int t = gen[i];
                if (t >= 0 && t < g_molly.vocab->vocab_size)
                    printf("%s ", g_molly.vocab->tokens[t]);
            }
            printf("\n"); fflush(stdout);

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

            pthread_mutex_lock(&g_molly_lock);
            int n = (gn > 32) ? 32 : gn;
            memcpy(g_molly.last_tokens, gen, (size_t)n * sizeof(int));
            g_molly.last_len = n;
            g_molly.utter_count++;
            pthread_mutex_unlock(&g_molly_lock);
        }

        usleep(MOLLY_TICK_USEC);
        fflush(stdout);
    }
    return NULL;
}

/* Conservative cooccur-only blend. Earlier emerged+vocab+Hebbian copy
 * was destabilising Trinity on Linux glibc malloc; bisected to this
 * function. Now blends only the base-glyph cooccur cells; emerged
 * regrows from substrate via adaptive emerge in main file. */
static int blend_affair_surface(const Cave* mate, Cave* child) {
    if (!mate || !child || !child->model || !child->vocab) return -1;
    if (!g_molly.model || !g_molly.vocab) return -1;

    CaveModel* ma = mate->model;
    CaveModel* mb = g_molly.model;
    CaveModel* mc = child->model;

    int B = child->vocab->base_size;
    if (B > COOCCUR_SIZE) B = COOCCUR_SIZE;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < B; j++) {
            mc->cooccur.matrix[i][j] =
                0.5f * (ma->cooccur.matrix[i][j] + mb->cooccur.matrix[i][j]);
            mc->cooccur.pair_count[i][j] =
                (ma->cooccur.pair_count[i][j] + mb->cooccur.pair_count[i][j]) / 2;
        }
    }
    mc->cooccur.total_interactions =
        (ma->cooccur.total_interactions + mb->cooccur.total_interactions) / 2;
    mc->cooccur.last_emergence = mc->cooccur.total_interactions;
    return 0;
}
