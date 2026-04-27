/*
 * cavellman_42.h — shared types + cross-module API for the 42 fork.
 *
 * cavellman_42.c keeps the bulk of the engine (model load/forward/sample,
 * ring async + learner + family mitosis + Trinity affair). predator.c
 * is extracted as the first real module-split: god-wrath storms with
 * multi-victim affair, cooccur siphon, permanent scar.
 *
 * If you add a new module: include this header, declare your statics,
 * extern the globals you need from cavellman_42.c, and add the function
 * prototypes here for cross-module calls.
 */
#ifndef CAVELLMAN_42_H
#define CAVELLMAN_42_H

#include "notorch.h"
#include <stdint.h>
#include <pthread.h>
#include <sys/types.h>

/* ── Limits (mirrored from cavellman_42.c — single source) ─────────────── */
#define MAX_VOCAB        256
#define MAX_SEQ          128
#define MAX_EMERGED      128
#define COOCCUR_SIZE     256
#define HEBBIAN_RANK     4
#define EMERGE_THRESHOLD 0.75f
#define DISCIPLINE_WINDOW 50

#define SURVIVAL_USES    5
#define SURVIVAL_WINDOW  500
#define MAX_DEPTH        5

#define TUNNEL_THRESHOLD    0.40f
#define EXCITEMENT_DECAY    0.94f
#define DISSONANCE_DECAY    0.90f
#define EXCITEMENT_CAP      2.5f
#define MATURITY_WINDOW     40
#define MATURITY_STEP       0.005f
#define MATURITY_CAP        0.30f
#define SPEAK_RATIO_HIGH    0.70f
#define SPEAK_RATIO_LOW     0.20f
#define DUAL_TICK_US        100000
#define DUAL_MAX_GEN        24

#define MICRO_MIN_BYTES     2500
#define MICRO_MIN_NOVELTY   8.0f
#define MICRO_MIN_RESONANCE 15.0f
#define MICRO_TRAIN_STEPS   "300"

#define COLONY_MAX          8
#define DNA_DIR             "dna"
#define DNA_MAX_AGE         3600

#define MITOSIS_COOLDOWN_TICKS  500
#define NEWBORN_IMMUNITY_TICKS  800

/* ── Types (shared across modules) ─────────────────────────────────────── */

typedef struct {
    char tokens[MAX_VOCAB][32];
    int  vocab_size;
    int  base_size;
    int  bos_id;
    int  mask_id;
} CaveVocab;

typedef struct {
    float matrix[COOCCUR_SIZE][COOCCUR_SIZE];
    int   pair_count[COOCCUR_SIZE][COOCCUR_SIZE];
    int   total_interactions;
    int   last_emergence;
} CoOccurrence;

typedef struct {
    int   glyph_a;
    int   glyph_b;
    float strength;
    int   born_at;
    int   use_count;
    int   alive;
    int   depth;
    char  name[32];
} EmergedSymbol;

typedef struct {
    nt_tensor *rms1, *wq, *wk, *wv, *wo;
    nt_tensor *rms2, *w_fc1, *w_fc2;
    float *heb_A_q, *heb_B_q;
    float *heb_A_v, *heb_B_v;
} Layer;

typedef struct {
    nt_tensor* wte;
    nt_tensor* wpe;
    Layer*     layers;
    nt_tensor* rms_f;
    nt_tensor* head;
    int E, H, HD, FFN_D, N_L, CTX;
    CoOccurrence cooccur;
    EmergedSymbol emerged[MAX_EMERGED];
    int n_emerged;
    float hebbian_lr;
    float hebbian_decay;
} CaveModel;

typedef struct {
    float excitement;
    float coherence_floor;
    float baseline_floor;
    float dissonance;
    int   spoke_count;
    int   total_count;
    const char* name;

    long  mass_bytes;
    float mass_novelty;
    float mass_resonance;
    char  holding_path[512];
    char  weights_path[512];
    char  next_weights_path[512];
    const char* preset_name;
    int   microtrain_active;
    pid_t microtrain_pid;
    int   microtrain_done_count;

    int   immunity_ticks;
} CaveField;

typedef struct {
    CaveModel* model;
    CaveVocab* vocab;
    CaveField  field;
    int        owns_vocab;
    int        is_founder;
    int        is_lover;
    int        is_bastard;
    int     last_tokens[32];
    int     last_len;
    int64_t spore_saved_at;
} Cave;

typedef struct {
    pthread_t  thread;
    int        started;
    CaveModel* model;
    CaveVocab* vocab;
    char       weights_path[512];
    int        last_tokens[32];
    int        last_len;
    int        utter_count;
} MollyState;

typedef struct {
    CaveModel* model;
    CaveVocab* vocab;
    char       weights_path[512];
    int        loaded;
    int        visit_count;
} PredatorState;

/* ── Globals (defined in cavellman_42.c, used by predator.c) ────────────── */
extern Cave* g_colony[COLONY_MAX];
extern int   g_colony_n;
extern int   g_children_born;
extern int   g_mitosis_cooldown;

extern MollyState     g_molly;
extern PredatorState  g_predator;
extern int   g_predator_storm_ticks_left;
extern int   g_predator_storm_duration;
extern float g_predator_strike_prob;
extern int   g_predator_total_strikes;

/* ── Cross-module function prototypes ──────────────────────────────────── */
int   blend_weights(const char* a_path, const char* b_path, const char* out_path);
int   copy_file(const char* src, const char* dst);
Cave* cave_new(const char* name, float baseline_floor,
               const char* weights_path, const char* preset_name);
int   colony_add(Cave* c);

/* Predator module API (defined in predator.c, called from orchestrator). */
void  try_predator_storm(const char* preset_name);

#endif /* CAVELLMAN_42_H */
