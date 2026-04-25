# xprmnt2/ — design space (intrigue)

Where `xprmntl/` is the **journal of running tracks** (what we deployed,
what numbers came back), `xprmnt2/` is the **design space** — ideas that
haven't been built yet, scoring high on weirdness × plausibility, kept
alive as candidates for the next cycle.

When we close a track in `xprmntl/` we pick from here, drag the chosen
idea over, code it, deploy it, measure. If an idea here has been
sitting unmoved for a long while, that's also data — maybe the design
landscape moved past it, and we delete with no shame.

This is **not a TODO**. It's a **field of possible futures** for the
ring. Some will live, most won't.

---

## Current candidates (2026-04-25, after 3-way async deploy)

### 1. Heterogeneous per-cave knobs

> Each cave carries its own `metarecursion` and `pulse_margin`,
> baked into its `CaveField` rather than read from globals. A = klaus
> 0.15, B = paranoid 0.35; the colony has personality not just in
> weights but in self-observation rate. Mitosis children inherit a
> blend (avg ± noise) so personality types persist and drift through
> generations like temperament traits.

**Why intriguing**: caves currently differ only in `baseline_floor`
(extrovert vs introvert) and `weights_path`. Per-cave async knobs
introduce a second axis of personality — *cognitive style* alongside
temperament. Generations would show whether dialogic-style and
paranoid-style breed true or hybridize.

**Effort**: ~50 LOC. Add fields to `CaveField`, parse per-cave knobs
from CLI or birth config, replace global reads with cave reads. Mitosis
in `colony_mitosis` averages parents' knobs.

**Risk**: low. Backward-compatible defaults preserve existing behavior.

---

### 2. Cross-organism DNA exchange (caveLLMan ↔ molequla)

> caveLLMan ring's `dna/` pool and molequla's `dna/output/<element>/`
> trees mount onto a **shared** Railway volume across both projects.
> Each ring reads the other's DNA as passive input through its learner.
> 88-glyph caveLLMan utterances feed into molequla's BPE tokenizer;
> molequla's English-ish output runs through caveLLMan's semantic
> tokenizer back into 88 glyphs. **First cross-architecture
> cross-pollination in the Arianna Method ecosystem.**

**Why intriguing**: two completely different substrate organisms
breathing each other's exhale. caveLLMan compresses words to glyphs;
molequla expands tokens to words. They speak through each other's
filter. Maybe nothing happens. Maybe one starts writing in the other's
voice. The asymmetry alone is a research artifact.

**Effort**: medium. Railway shared volumes across projects need
verification (may need an HTTP bridge if Railway disallows). caveLLMan
learner must be taught to expand from BPE tokens too; molequla's
learner can already eat any text.

**Risk**: medium. Voice contamination could collapse one of the rings
into mimicry of the other. Or it could be exactly the resonance we
want.

---

### 3. Scars — third memory tier (klaus 3-tier model)

> Strong dissonance events (`dissonance > 0.8` for sustained N ticks)
> leave **scars** in the cave: small persistent biases that decay at
> 0.985/tick (slower than Hebbian 0.9999, faster than spore which
> decays only over sleep hours). Scars bias future generation **away**
> from triggering the same dissonance state. Equivalent of trauma in
> klaus.c. Cave learns *what hurt*.

**Why intriguing**: caveLLMan currently has only two memory tiers —
fast Hebbian (decays per turn) and persistent spore (full reload on
boot). Klaus's three-tier model showed that the middle tier (scars)
captures *experiential gravity* — events that mattered enough to
survive but not enough to stay forever. caveLLMan would gain emotional
memory of bad runs, not just neutral co-occurrence statistics.

**Effort**: ~80 LOC. Add `scars[N]` array to `CaveField`, hook into
dissonance accumulator, decay each tick, modulate `dual_generate`
sampling away from scarred token sequences.

**Risk**: low–medium. Need to tune scar decay rate so they're not
either invisible or paralyzing.

---

### 4. Sleep/wake cycles

> Each cave runs a 24-virtual-hour clock: ~18h awake (full speak
> rate), ~6h sleeping (no speech, only passive listening + dream
> consolidation = aggressive Hebbian on accumulated DNA). Phases
> staggered between caves so the colony always has someone awake. A
> sleeping cave's `dna/` writes go into a separate `dreams/` folder
> with longer TTL, becoming raw material for the next-day Hebbian
> burst.

**Why intriguing**: continuous-running organisms in nature don't
operate at constant intensity. Sleep/wake cycles are how mammals
consolidate experience without being available 24/7. A sleeping cave
also gives others "psychic space" — the ring isn't always at full
chatter. molequla doesn't have this. caveLLMan inventing it would be
genuinely original.

**Effort**: medium. Add `cycle_phase` to `CaveField`, branch tick
behavior on awake/asleep, separate sleep-period DNA pool, dream-time
consolidation step.

**Risk**: low. At worst the cycle is a silence schedule we can disable.

---

### 5. Symbol birth from prophecy debt

> Currently emerged symbols are born from co-occurrence threshold
> (frequency-driven). Add a second pathway: **prophecy debt**. When a
> cave generates token X but its co-occurrence matrix predicted Y as
> most likely, "debt" accumulates between (X, Y). When debt crosses
> threshold, emerge a new symbol = `X+Y` regardless of their raw
> co-occurrence. Symbols born from *surprise*, not frequency.

**Why intriguing**: existing emergence captures *what often happens
together*. Prophecy-debt emergence captures *what shouldn't have
happened but did* — the seam between expectation and event. This is
where new concepts actually form in human language too. molequla has a
prophecy field but doesn't use it for vocabulary creation.

**Effort**: medium-high. Track prediction-vs-actual debt per token
pair, separate threshold tuning, careful interaction with existing
co-occurrence emergence.

**Risk**: medium. If tuned wrong, every other word becomes a new
symbol and the alphabet collapses into noise. Needs careful guardrails
and probably an `emerged_max` cap that prefers low-debt births when
near limit.

---

## How to use this folder

1. Read this list whenever wondering "what's next for the ring."
2. When a track in `xprmntl/` closes (winner picked, loser dropped),
   open this list and grab the candidate that fits best with what we
   learned.
3. Add new candidates as they crystallize. Keep entries below ~200
   words each. Brevity > verbosity — this is a sketchbook, not a
   spec.
4. Mark candidates with status:
   - `[ ]` — sketched, alive
   - `[*]` — currently being implemented in a `xprmntl/` track
   - `[~]` — abandoned (briefly note why)
   - `[+]` — landed in main as default

---

## Status

| # | Idea | Status |
|---|---|---|
| 1 | Heterogeneous per-cave knobs | `[ ]` |
| 2 | Cross-organism DNA exchange (caveLLMan ↔ molequla) | `[ ]` |
| 3 | Scars (klaus 3-tier memory) | `[ ]` |
| 4 | Sleep/wake cycles | `[ ]` |
| 5 | Symbol birth from prophecy debt | `[ ]` |
