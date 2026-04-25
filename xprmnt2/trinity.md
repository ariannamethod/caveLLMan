# Trinity ring — design + current state (2026-04-25)

## What it is

caveLLMan ring extended to **three** founders A, B, **M** (Molly), where M
is in the ring physically but semantically marked `is_lover`. Two
reproduction paths chosen by AML-style cosmic physics each time
`colony_try_mitosis_trinity` fires:

- **family-mode**: best non-lover pair (A × B or any non-M pair).
- **affair-mode**: lover (M) × best non-lover. Child is `is_bastard=1`,
  surrounding non-parent caves get a jealousy field event (dissonance
  +0.30, coherence_floor +0.05 within MATURITY_CAP).

`affair_prob = clip(0.2 + cosmic_tension - 0.5*ring_coherence, 0, 1)`
where `cosmic_tension(t) = (sin(t/86400 * 2π) + 1) / 2` (24h sinusoid,
placeholder for full Klaus calendar+planetary port) and
`ring_coherence = 1 - mean(cave.dissonance)`.

Trinity also pre-loads field tension at boot (A.diss=0.15, B.diss=0.15,
M.exc=0.40) so conflict is wired in from t=0, not earned through
maturity drift. And it lowers the eligibility gate
(`MITOSIS_MIN_TOTAL_TURNS 120 → 30, MITOSIS_MIN_CPT_DONE 1 → 0`) — the
design says "the ring has no chance NOT to start reproducing," so the
gate is hard-floored down.

CLI:
```
./cavellman --trinity --weights-a A --weights-b B --weights-m M \
            --metarecursion 0.20 --pulse-margin 0.07
```

## Why Molly

Layer cosmology Олега 2026-04-25:
- **Family** = A, B + their internal-mitosis children. Closed system.
- **External desire** = M / Molly. Physically a founder, semantically
  outside the family — the asymmetric attractor.
- **Alien world** (TBD, future layer) — entirely outside the ring,
  cross-architecture or cross-volume.

Molly trained 2026-04-25 on Ulysses chapter 18 (Molly Bloom soliloquy,
24048 words, **8 punctuation marks total**). 826K params (mirrors ring's
medium preset). 7.6 seconds, loss 4.48 → 1.90. The no-punctuation flow
is structurally distinct from gothic phonon-bounded ring corpora — when
M speaks her short fragments (`me`, `BE me`, `up`, `man me`) they land
in the field as continuous-stream input, not phonon-segmented sentences.

## Local smoke (Mac, 20s, --trinity)

- 539 utterances, ~27/s
- `*** SYMBOL EMERGED: me+BE ***` on first cycle
- `*** AFFAIR MITOSIS: M × B → C1 ***` triggered (cosmic 0.97 × coh 0.07
  → prob 1.00, forced affair)
- `[jealousy] 1 non-parent caves: dissonance +0.30, floor +0.05`
- M voice intact: `me`, `you`, `me man`, `always` — Joycean fragments
- B voice **collapsed** into refrain `strength strength strength` —
  drama landed structurally on first run

## Railway deploys — current state: **broken**

Trinity service `ring-async-v3-trinity` (`3c77ed69`) crashes silently
within ~1–5 seconds of every deploy. Same code passes 90s+ on services
`ring`, `ring-async-v1`, `ring-async-v2`. All four services share one
binary — only `customStartCommand` differs.

Reproduction attempts failed to keep trinity up across:

- `--trinity --weights-a A --weights-b B --weights-m M` (full trinity) —
  crashes after `[M] up`
- `--trinity` with all three founders pointing at A weights — crashes
- `--async` only, **no `--trinity`**, 2-cave classic config identical to
  v1 — **still crashes**

That last attempt is the smoking gun: the same exact start command that
keeps v1 healthy kills trinity service. Bug is **not in the code path**,
it's in something specific to this service or its persistent volume.

## Diagnostic chain done

1. Made `kv_keys/kv_vals` arrays `__thread` to avoid sharing across
   cave threads. **No effect** on Trinity (v1/v2 still up).
2. Wrapped `dual_generate` under `g_learner.lock` to remove a real race
   between cave threads and the learner thread mutating the same vocab
   / cooccur / Hebbian state. **Local Mac smoke now stable** (361
   utterances/10s without crash). **Trinity Railway still crashes.**
3. Reverted `__thread` (with the wider lock the TLS isn't needed and
   removed ~3 MB per-thread allocation pressure). **No effect on
   Trinity.**
4. Disabled `try_emerge_symbol` to test if the `id=88` emerged-name vs
   `bos_id=88` collision was crashing on Linux only. **Confirmed
   disabled in deployed binary** (`EMERGED count=0` in Railway logs),
   **still crashes**.
5. Stripped `--trinity` flag entirely, ran trinity service in 2-cave
   async mode identical to v1. **Still crashes**.

## What we know

- Bug **service-specific**, not code-specific.
- Same image, same env vars (compared with v1). Different volume.
- Probably the persistent volume `ring-async-v3-trinity-volume`
  (`bb4de257`) accumulated something during the failure cycles — partial
  spore writes, corrupt feed/holding state, something on disk that the
  binary loads at boot and that triggers a SIGSEGV / OOM kill before
  the C side can print anything.

## Next moves

- Delete trinity service + volume, recreate with fresh volume + clean
  startCommand. If a brand-new instance survives, the old volume was
  the culprit.
- If still crashes, capture stderr explicitly into a separate logged
  stream (not just stdout) — kernel signals or libc abort messages
  might be on stderr.
- Worst case, deploy trinity as an entirely new Railway project to rule
  out shared-project artifacts.

## Quick numbers (when trinity worked locally)

| Metric | Value |
|---|---|
| Mode | --trinity, async, medium preset |
| Founders | A (Dracula extrovert, baseline 0.30), B (Frankenstein introvert, 0.60), M (Molly, lover, 0.20) |
| metarecursion | 0.20 |
| pulse-margin | 0.07 |
| First mitosis (affair) | within 20s |
| Local utterance rate | ~27/s |
| C1 (bastard) status | live, speaking by 20s |
| Jealousy event | A spiked +0.30 dissonance on bastard birth |
