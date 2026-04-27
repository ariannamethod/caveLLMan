/*
 * predator.c — H, божья кара. Event-driven storm engine, no thread.
 *
 * Loaded once via --predator <weights>. Per orchestrator tick, with
 * probability g_predator_strike_prob, a "storm" descends:
 *   - dissonance + excitement spike on ALL caves
 *   - top-K victims (escalating 2..4 per strike count) get forced
 *     predator-affair: H × victim → P{n} bastard
 *   - cooccur siphon: H absorbs 0.20 × victim cooccur (cap 1.0)
 *   - permanent scar: victim coherence_floor += 0.05
 * Storm decays after duration ticks, scars stay forever.
 *
 * First real module-split out of cavellman_42.c — this file is
 * self-contained: types live in cavellman_42.h, state is exported to
 * the rest of the engine via extern in the header.
 */
#include "cavellman_42.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

/* Predator state — definitions live here, extern'd from cavellman_42.h. */
PredatorState g_predator = {0};
int   g_predator_storm_ticks_left = 0;
int   g_predator_storm_duration   = 60;     /* ~6 sec at 0.1s tick */
float g_predator_strike_prob      = 0.005f; /* ~0.5%/sec on 1Hz orch */
int   g_predator_total_strikes    = 0;

/* Cave-with-Predator affair: same blend mechanics as Molly's affair but
 * with g_predator's stable on-disk weights. Bastard names P{n}. */
static Cave* colony_affair_with_predator(int mate_idx, const char* preset_name) {
    if (mate_idx < 0 || mate_idx >= g_colony_n) return NULL;
    if (g_colony_n >= COLONY_MAX) return NULL;
    if (!g_predator.loaded || !g_predator.model) return NULL;

    Cave* pa = g_colony[mate_idx];

    char child_name[16];
    snprintf(child_name, sizeof(child_name), "P%d", g_predator.visit_count + 1);

    char child_w[512], child_v[512], child_j[512];
    char parent_v[512], parent_j[512];
    snprintf(child_w, sizeof(child_w), "weights/cavellman_predator_%s.bin", child_name);
    snprintf(child_v, sizeof(child_v), "%s.vocab", child_w);
    snprintf(child_j, sizeof(child_j), "%s.json",  child_w);
    snprintf(parent_v, sizeof(parent_v), "%s.vocab", pa->field.weights_path);
    snprintf(parent_j, sizeof(parent_j), "%s.json",  pa->field.weights_path);

    if (blend_weights(pa->field.weights_path, g_predator.weights_path, child_w) != 0) {
        printf("  [strike] blend_weights failed for %s × Predator → %s\n",
               pa->field.name, child_w);
        return NULL;
    }
    copy_file(parent_v, child_v);
    copy_file(parent_j, child_j);

    float child_baseline = 0.5f * (pa->field.baseline_floor + 0.20f);
    char* permanent_name = strdup(child_name);
    Cave* child = cave_new(permanent_name, child_baseline, child_w, preset_name);
    if (!child) { free(permanent_name); return NULL; }

    int n = pa->last_len;
    if (n > 32) n = 32;
    if (n > 0) memcpy(child->last_tokens, pa->last_tokens, (size_t)n * sizeof(int));
    child->last_len = n;
    child->spore_saved_at = (int64_t)time(NULL);
    child->field.immunity_ticks = NEWBORN_IMMUNITY_TICKS;
    child->is_bastard = 1;

    colony_add(child);
    g_children_born++;
    g_predator.visit_count++;
    return child;
}

/* Memetic theft — H absorbs SIPHON_FRACTION of victim's cooccur, cap 1.0. */
#define SIPHON_FRACTION 0.20f
static void predator_siphon(const Cave* victim) {
    if (!g_predator.loaded || !g_predator.model) return;
    CoOccurrence* H = &g_predator.model->cooccur;
    const CoOccurrence* V = &victim->model->cooccur;
    int B = victim->vocab->base_size;
    if (B > COOCCUR_SIZE) B = COOCCUR_SIZE;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < B; j++) {
            float v = SIPHON_FRACTION * V->matrix[i][j];
            float h = H->matrix[i][j] + v;
            if (h > 1.0f) h = 1.0f;
            H->matrix[i][j] = h;
        }
    }
    H->total_interactions += V->total_interactions / 5;
}

/* Storm engine — invoked every orchestrator tick. Stochastic strike. */
#define PREDATOR_DISSONANCE_SPIKE 0.7f
#define PREDATOR_EXCITEMENT_SPIKE 0.4f
#define PREDATOR_SCAR_FLOOR_BUMP  0.05f
void try_predator_storm(const char* preset_name) {
    if (!g_predator.loaded) return;

    if (g_predator_storm_ticks_left > 0) {
        g_predator_storm_ticks_left--;
        if (g_predator_storm_ticks_left == 0) {
            printf("\n  *** PREDATOR STORM ENDS — physics returning to baseline ***\n\n");
        }
        return;
    }

    float roll = (float)rand() / (float)RAND_MAX;
    if (roll > g_predator_strike_prob) return;
    if (g_colony_n < 1) return;

    g_predator_total_strikes++;

    int n_victims = 2 + (g_predator_total_strikes / 3);
    if (n_victims > 4) n_victims = 4;
    if (n_victims > g_colony_n) n_victims = g_colony_n;

    int picked[16] = {0};
    int npicked = 0;
    for (int k = 0; k < n_victims && npicked < g_colony_n; k++) {
        int best = -1;
        float best_score = -1.0f;
        for (int ci = 0; ci < g_colony_n; ci++) {
            int already = 0;
            for (int p = 0; p < npicked; p++)
                if (picked[p] == ci) { already = 1; break; }
            if (already) continue;
            float s = g_colony[ci]->field.excitement +
                      0.5f * g_colony[ci]->field.dissonance;
            if (s > best_score) { best_score = s; best = ci; }
        }
        if (best < 0) break;
        picked[npicked++] = best;
    }

    printf("\n  *** PREDATOR STORM #%d — H descends, %d victims, all caves shudder ***\n",
           g_predator_total_strikes, npicked);

    for (int ci = 0; ci < g_colony_n; ci++) {
        Cave* c = g_colony[ci];
        c->field.dissonance += PREDATOR_DISSONANCE_SPIKE;
        if (c->field.dissonance > 1.0f) c->field.dissonance = 1.0f;
        c->field.excitement += PREDATOR_EXCITEMENT_SPIKE;
        if (c->field.excitement > EXCITEMENT_CAP) c->field.excitement = EXCITEMENT_CAP;
    }

    for (int p = 0; p < npicked; p++) {
        int vi = picked[p];
        Cave* v = g_colony[vi];

        predator_siphon(v);

        float ceil = v->field.baseline_floor + MATURITY_CAP;
        float new_floor = v->field.coherence_floor + PREDATOR_SCAR_FLOOR_BUMP;
        if (new_floor > ceil) new_floor = ceil;
        v->field.coherence_floor = new_floor;

        if (g_colony_n < COLONY_MAX) {
            Cave* bastard = colony_affair_with_predator(vi, preset_name);
            if (bastard) {
                printf("  *** PREDATOR AFFAIR: H × %s → %s (forced, scarred, siphoned) ***\n",
                       v->field.name, bastard->field.name);
            }
        }
    }

    g_predator_storm_ticks_left = g_predator_storm_duration;
    g_mitosis_cooldown = MITOSIS_COOLDOWN_TICKS;
}
