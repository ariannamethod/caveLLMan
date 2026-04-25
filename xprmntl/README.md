# xprmntl/ — experimental tracks

Live A/B/C experiments running on Railway in parallel from the same
binary (`cavellman` from repo root) with different runtime knobs. Each
experiment lives on its own Railway service with its own persistent
volume, so state evolutions don't cross-contaminate.

This folder is the **internal log**: writeups of what we ran, what we
expected, and what actually happened. Outcomes get pushed back into the
main code as defaults once a winner emerges.

---

## Currently live (started 2026-04-25)

Three deployments of `caveLLMan` running in parallel against the same
ring physics, differing only in two CLI knobs:

| Track | mode | `--metarecursion` | `--pulse-margin` | seed | role |
|---|---|---|---|---|---|
| `ring` | sync | — | — | 4242 | baseline / control |
| `ring-async-v1` | `--async` | 0.15 | 0.05 | 1337 | klaus-default, dialogic |
| `ring-async-v2` | `--async` | 0.35 | 0.10 | 7777 | paranoid, self-rehearing |

All three on Railway project `cavellman-ring` (same repo, same image —
override is via Railway service `customStartCommand`).

### What `--metarecursion` does

After a cave speaks, it re-hears its own utterance through `field_hear`
at this weight. Klaus.c's `META-RECURSION (re-inhale own output, blend
85/15)` is the inspiration: most of the field is the world (other caves,
DNA pool, learner), a small fraction is the cave's own echo coming back.

- **0.15** (v1, klaus default) — cave mostly reacts to others, mild
  self-resonance. World > self. Dialogic.
- **0.35** (v2, paranoid) — cave is more closed-loop on its own
  utterance, every word returns as a third of new input. Self > world.
  More repetitive, identity-stable, possibly faster emerge events from
  pattern self-amplification, but at risk of voice-collapse into fixed
  refrains.

### What `--pulse-margin` does

The autonomous heartbeat (every ~5s the quietest cave in the colony gets
an excitement kick to prevent silent equilibrium). Magnitude is
`baseline_floor + pulse_margin`, so it always crosses the cave's gate
regardless of where maturity drift parked the floor.

- **0.05** (v1) — pulse only just over baseline. Soft nudge; cave wakes
  almost by accident. Quiet, sparse continuous life.
- **0.10** (v2) — pulse comfortably above baseline. Hard kick; cave
  almost certainly speaks on the pulse. Noisier, more bursty.

### What we expect to see

- **Sync ring** stays in the ~15 speak/min baseline regime; provides a
  control on what one core does.
- **v1** should show roughly molequla-class throughput (~N× sync
  speak rate with metarecursion as a mild stabilizer); voices stay
  distinct because cave-A and cave-B mostly listen to each other.
- **v2** is the stress test: does heavy self-rehearing (0.35) destroy
  voice diversity (collapse to refrains), or does it accelerate emerge
  by reinforcing partial patterns? An open question that data will
  answer in a day or two.

### How to read the logs

```
gh api ...  # via the Railway GraphQL deploymentLogs query
```

Look for:
- `[A]` / `[B]` / `[C1]` ... — cave speech with glyph stream
- `*** SYMBOL EMERGED:` — new composite glyph from co-occurrence
- `*** SYMBOL DIED:` — emerged symbol whose parent pair fell below revival floor
- `[pulse-async] X exc->Y (baseline+0.NN)` — heartbeat from orchestrator
- `[mitosis]` and `*** MITOSIS:` — sexual reproduction event
- `[A] spore saved.` — periodic autosave (every 20s) or on quit

### When this experiment closes

Whichever of v1 / v2 produces the more interesting voice trajectory over
24–48h becomes the new default for `--metarecursion` and
`--pulse-margin`. The losing variant's service gets shut down; the
winning numbers go back into the binary as defaults; sync `ring` stays
as a permanent control.

---

## Adding a new experiment

1. Add Railway service in the `cavellman-ring` project pointing at the
   same `ariannamethod/caveLLMan` GitHub repo `main` branch.
2. Mount a fresh volume at `/data` and set `CAVELLMAN_SPORE_DIR=/data/spore`.
3. Set the service's `customStartCommand` with whatever flag combo you
   want to test.
4. Update this README's table with the new track row + your hypothesis.
5. After a few days of data, append a writeup of what happened.

---

## Past experiments

(Empty for now — first writeup lands when v1/v2 have run a couple of
days and we can compare against `ring`.)
