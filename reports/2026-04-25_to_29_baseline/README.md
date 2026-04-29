# caveLLMan ring — Hebbian-only baseline 2026-04-25 to 2026-04-29

Pre-recipe baseline run on Railway (`cavellman-ring` project,
five services live during this window).

## Why this archive exists

Investigation 2026-04-29 (task #17) found that the running services
had two issues:

1. The ring spawns microtraining children via `execlp("./train_cavellman", …)`
   (`cavellman.c:2211`, `42/cavellman_42.c:1693`), but the
   Dockerfile only ran `make cavellman` — not `make train_cavellman`.
   Every microtrain child therefore exited 127 ("command not found").
   v4-storm logs showed 17 status=127 in 200 lines (vs 0-4 elsewhere)
   simply because storm has the most cave threads → most spawns.

2. `OPENBLAS_NUM_THREADS=1` had been added to the Dockerfile back on
   2026-04-25 to fix the SIGSEGV race in concurrent sgemv, so half the
   Railway CPU-fix recipe was already in place. The other half — the
   compile-time flags `-O3 -march=native -mtune=native` — was not.

Net effect during this window: Hebbian / cooccur / emerged / mass
adaptation worked, but gradient microtraining did not. The five
services ran as a **pure-Hebbian ecosystem**.

## After this archive

`main` is patched (merge commit `89170e4`) to:
- `make cavellman train_cavellman` in Dockerfile so the binary exists.
- `-O3 -march=native -mtune=native` in Makefile so the recipe is
  complete (Henry session 2026-04-29 measured 7.4× end-to-end speedup
  on a similar workload).

Auto-deploy on main push triggered redeploy of all main-tracking
services. The /data/spore volume persists Hebbian state across the
restart — cooccur / emerged / hebbian / last_tokens / saved_at carry
forward; in-memory transient (~last 100 ticks per cave) is lost.

Two services dropped: `ring-async-v1` (klaus-default 0.15 / 0.05) and
`ring-async-v2` (paranoid 0.35 / 0.10) — knob variants on metarecursion,
less character-distinct than the kept three. Also dropped
`ring-cpufix` — it became a duplicate of `ring` after the recipe
landed on main.

Three services remain after redeploy:
- `ring` (sync 2-cave baseline, `--preset medium --seed 4242`)
- `ring-async-v3-trinity` (Molly + cosmic-physics affair + jealousy)
- `ring-async-v4-storm` (predator engine, `cavellman_42` fork)

The continuation run is the «with-recipe + working microtrain» half
of the before/after pair.

## Files

- `ring_logs_tail_5000.txt` — sync 2-cave baseline.
- `ring-async-v1_logs_tail_5000.txt` — klaus-default async (dropped).
- `ring-async-v2_logs_tail_5000.txt` — paranoid async (dropped).
- `ring-async-v3-trinity_logs_tail_5000.txt` — trinity async.
- `ring-async-v4-storm_logs_tail_5000.txt` — storm fork.

Each file is the last 5000 lines of the last successful deployment
prior to 2026-04-29T14:52Z (when the main-push redeploy triggered).
