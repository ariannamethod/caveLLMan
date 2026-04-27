# 42/ — gravitational dispatch fork

Real fork of `cavellman.c`, not a CLI flag. Lives as a separate binary
(`cavellman_42`) with a Makefile target and an independent Railway service
(planned: `ring-async-v5-42`). Production `cavellman.c` keeps stable;
this folder is for the radical pivot.

## What's different from cavellman.c (planned)

- **Gravitational dispatch.** Each cave has a 2D position + mass on a
  ring plane. `mass = α·total_count + β·speak_ratio + γ·microtrain_done +
  δ·hebbian_norm`. Pairwise force `F_ij = G·m_i·m_j / r²_ij` with a
  softening floor. Caves drift under summed forces. **Permission to
  speak** is a function of local gravitational potential well depth (a
  cave at the bottom of a deep well speaks more often; a cave on a
  flat plane stays silent). The silence-gate from cavellman.c is gone.

- **Mitosis by Roche limit.** Two caves whose orbital distance falls
  below `R_roche(m_i, m_j)` for sustained N ticks merge into a child
  whose surface state is the gravitational mean of theirs. Family /
  affair / predator distinction collapses into one continuum: it's
  always *who's nearby*.

- **Cosmic_tension stays** but as a global modulator of `G` rather than
  affair_prob — gravity itself breathes on the 24h sin (and longer
  seasonal sin once added).

- **Ring physics first, transformer second.** Same model load path
  (preset shape detection, async threads, KV cache under
  `g_learner.lock`), but the speak gate is replaced wholesale.

This file is the starting point — copy of stable `cavellman.c` from the
day 42/ was opened. Each subsequent commit on this folder either
deletes a cavellman.c-mode behaviour or adds a gravitational one. When
all the original speak/mitosis/affair gates are removed and replaced,
the fork is "complete enough" to deploy as `ring-async-v5-42`.

## Build

```
make cavellman_42         # adds in caveLLMan root Makefile
```

## Why a real fork, not a flag

cavellman.c is a stable monolith with four production startCommand modes
(sync / async-v1 / async-v2 / trinity / trinity+predator). Wedging a
fundamentally different physics inside it via a `--gravity` flag would
double the surface area of every existing path. Cleaner: copy, mutate,
ship as a sibling. xprmnt2/ is design space; 42/ is a real radical
prototype.
