# caveLLMan

### 88 hieroglyphs. any English text. one shared alphabet.

*30,000 years ago, humans drew 32 recurring signs across 146 cave sites on four continents. We added 56 more for the 21st century — and built a transformer that compresses English into them, a pair that talks to each other through them, and a runtime that keeps training itself on what it hears.*

---

## What is this?

caveLLMan is a transformer that compresses English text into 88 hieroglyphic concepts. Feed it Dracula, news articles, or code documentation — the **semantic tokenizer** maps every English word to one of 88 universal symbols, and the model learns patterns in this compressed space. Multilingual tokenization is planned once the engine is language-agnostic end-to-end.

```
"the sun rose and the birds started singing"  →  light tree and animal before music
"Count Dracula stood in the dark castle"      →  dark stone and wait man
"she wrote code all night and found the bug"  →  woman AI dark and make light
```

Two training modes:
- **Diffusion** — randomly masks positions, trains bidirectional prediction
- **Autoregressive** — standard left-to-right next-token prediction

At runtime there is only one mode: **two caves talk**. A single-engine dialogue loop with a human at the center was deprecated — the human is no longer required to be present, let alone central. No Python. No pip. No torch. C engine built on [notorch](https://github.com/ariannamethod/notorch).

---

## The 88 Hieroglyphs

<table>
<tr><th colspan="9">NATURE</th></tr>
<tr>
<td align="center"><img src="glyphs/water.svg" width="40"><br><sub>water</sub></td>
<td align="center"><img src="glyphs/fire.svg" width="40"><br><sub>fire</sub></td>
<td align="center"><img src="glyphs/earth.svg" width="40"><br><sub>earth</sub></td>
<td align="center"><img src="glyphs/stone.svg" width="40"><br><sub>stone</sub></td>
<td align="center"><img src="glyphs/tree.svg" width="40"><br><sub>tree</sub></td>
<td align="center"><img src="glyphs/sky.svg" width="40"><br><sub>sky</sub></td>
<td align="center"><img src="glyphs/light.svg" width="40"><br><sub>light</sub></td>
<td align="center"><img src="glyphs/dark.svg" width="40"><br><sub>dark</sub></td>
<td align="center"><img src="glyphs/cold.svg" width="40"><br><sub>cold</sub></td>
</tr>
<tr><th colspan="8">BEINGS</th></tr>
<tr>
<td align="center"><img src="glyphs/person.svg" width="40"><br><sub>person</sub></td>
<td align="center"><img src="glyphs/man.svg" width="40"><br><sub>man</sub></td>
<td align="center"><img src="glyphs/woman.svg" width="40"><br><sub>woman</sub></td>
<td align="center"><img src="glyphs/child.svg" width="40"><br><sub>child</sub></td>
<td align="center"><img src="glyphs/old.svg" width="40"><br><sub>old</sub></td>
<td align="center"><img src="glyphs/spirit.svg" width="40"><br><sub>spirit</sub></td>
<td align="center"><img src="glyphs/AI.svg" width="40"><br><sub>AI</sub></td>
<td align="center"><img src="glyphs/animal.svg" width="40"><br><sub>animal</sub></td>
</tr>
<tr><th colspan="5">BODY</th></tr>
<tr>
<td align="center"><img src="glyphs/body.svg" width="40"><br><sub>body</sub></td>
<td align="center"><img src="glyphs/food.svg" width="40"><br><sub>food</sub></td>
<td align="center"><img src="glyphs/sleep.svg" width="40"><br><sub>sleep</sub></td>
<td align="center"><img src="glyphs/pain.svg" width="40"><br><sub>pain</sub></td>
<td align="center"><img src="glyphs/strength.svg" width="40"><br><sub>strength</sub></td>
</tr>
<tr><th colspan="8">EMOTION</th></tr>
<tr>
<td align="center"><img src="glyphs/joy.svg" width="40"><br><sub>joy</sub></td>
<td align="center"><img src="glyphs/grief.svg" width="40"><br><sub>grief</sub></td>
<td align="center"><img src="glyphs/love.svg" width="40"><br><sub>love</sub></td>
<td align="center"><img src="glyphs/fear.svg" width="40"><br><sub>fear</sub></td>
<td align="center"><img src="glyphs/anger.svg" width="40"><br><sub>anger</sub></td>
<td align="center"><img src="glyphs/longing.svg" width="40"><br><sub>longing</sub></td>
<td align="center"><img src="glyphs/tired.svg" width="40"><br><sub>tired</sub></td>
<td align="center"><img src="glyphs/stress.svg" width="40"><br><sub>stress</sub></td>
</tr>
<tr><th colspan="11">VERBS</th></tr>
<tr>
<td align="center"><img src="glyphs/go.svg" width="40"><br><sub>go</sub></td>
<td align="center"><img src="glyphs/make.svg" width="40"><br><sub>make</sub></td>
<td align="center"><img src="glyphs/break.svg" width="40"><br><sub>break</sub></td>
<td align="center"><img src="glyphs/see.svg" width="40"><br><sub>see</sub></td>
<td align="center"><img src="glyphs/speak.svg" width="40"><br><sub>speak</sub></td>
<td align="center"><img src="glyphs/hear.svg" width="40"><br><sub>hear</sub></td>
<td align="center"><img src="glyphs/seek.svg" width="40"><br><sub>seek</sub></td>
<td align="center"><img src="glyphs/give.svg" width="40"><br><sub>give</sub></td>
<td align="center"><img src="glyphs/want.svg" width="40"><br><sub>want</sub></td>
<td align="center"><img src="glyphs/miss.svg" width="40"><br><sub>miss</sub></td>
<td align="center"><img src="glyphs/agree.svg" width="40"><br><sub>agree</sub></td>
</tr>
<tr><th colspan="6">SOCIAL</th></tr>
<tr>
<td align="center"><img src="glyphs/home.svg" width="40"><br><sub>home</sub></td>
<td align="center"><img src="glyphs/outside.svg" width="40"><br><sub>outside</sub></td>
<td align="center"><img src="glyphs/work.svg" width="40"><br><sub>work</sub></td>
<td align="center"><img src="glyphs/internet.svg" width="40"><br><sub>internet</sub></td>
<td align="center"><img src="glyphs/bond.svg" width="40"><br><sub>bond</sub></td>
<td align="center"><img src="glyphs/conflict.svg" width="40"><br><sub>conflict</sub></td>
</tr>
<tr><th colspan="6">MIND</th></tr>
<tr>
<td align="center"><img src="glyphs/know.svg" width="40"><br><sub>know</sub></td>
<td align="center"><img src="glyphs/idea.svg" width="40"><br><sub>idea</sub></td>
<td align="center"><img src="glyphs/think.svg" width="40"><br><sub>think</sub></td>
<td align="center"><img src="glyphs/dream.svg" width="40"><br><sub>dream</sub></td>
<td align="center"><img src="glyphs/remember.svg" width="40"><br><sub>remember</sub></td>
<td align="center"><img src="glyphs/lie.svg" width="40"><br><sub>lie</sub></td>
</tr>
<tr><th colspan="5">SPACE</th></tr>
<tr>
<td align="center"><img src="glyphs/path.svg" width="40"><br><sub>path</sub></td>
<td align="center"><img src="glyphs/up.svg" width="40"><br><sub>up</sub></td>
<td align="center"><img src="glyphs/down.svg" width="40"><br><sub>down</sub></td>
<td align="center"><img src="glyphs/far.svg" width="40"><br><sub>far</sub></td>
<td align="center"><img src="glyphs/back.svg" width="40"><br><sub>back</sub></td>
</tr>
<tr><th colspan="5">TIME</th></tr>
<tr>
<td align="center"><img src="glyphs/before.svg" width="40"><br><sub>before</sub></td>
<td align="center"><img src="glyphs/now.svg" width="40"><br><sub>now</sub></td>
<td align="center"><img src="glyphs/after.svg" width="40"><br><sub>after</sub></td>
<td align="center"><img src="glyphs/never.svg" width="40"><br><sub>never</sub></td>
<td align="center"><img src="glyphs/always.svg" width="40"><br><sub>always</sub></td>
</tr>
<tr><th colspan="8">GRAMMAR</th></tr>
<tr>
<td align="center"><img src="glyphs/not.svg" width="40"><br><sub>not</sub></td>
<td align="center"><img src="glyphs/many.svg" width="40"><br><sub>many</sub></td>
<td align="center"><img src="glyphs/much.svg" width="40"><br><sub>much</sub></td>
<td align="center"><img src="glyphs/and.svg" width="40"><br><sub>and</sub></td>
<td align="center"><img src="glyphs/one.svg" width="40"><br><sub>one</sub></td>
<td align="center"><img src="glyphs/question.svg" width="40"><br><sub>question</sub></td>
<td align="center"><img src="glyphs/how.svg" width="40"><br><sub>how</sub></td>
<td align="center"><img src="glyphs/cause.svg" width="40"><br><sub>cause</sub></td>
</tr>
<tr><th colspan="13">EXTENDED</th></tr>
<tr>
<td align="center"><img src="glyphs/me.svg" width="40"><br><sub>me</sub></td>
<td align="center"><img src="glyphs/you.svg" width="40"><br><sub>you</sub></td>
<td align="center"><img src="glyphs/other.svg" width="40"><br><sub>other</sub></td>
<td align="center"><img src="glyphs/money.svg" width="40"><br><sub>money</sub></td>
<td align="center"><img src="glyphs/change.svg" width="40"><br><sub>change</sub></td>
<td align="center"><img src="glyphs/write.svg" width="40"><br><sub>write</sub></td>
<td align="center"><img src="glyphs/choose.svg" width="40"><br><sub>choose</sub></td>
<td align="center"><img src="glyphs/help.svg" width="40"><br><sub>help</sub></td>
<td align="center"><img src="glyphs/have.svg" width="40"><br><sub>have</sub></td>
<td align="center"><img src="glyphs/free.svg" width="40"><br><sub>free</sub></td>
<td align="center"><img src="glyphs/death.svg" width="40"><br><sub>death</sub></td>
<td align="center"><img src="glyphs/music.svg" width="40"><br><sub>music</sub></td>
<td align="center"><img src="glyphs/good.svg" width="40"><br><sub>good</sub></td>
</tr>
<tr><th colspan="4">SCALE + SUPER</th></tr>
<tr>
<td align="center"><img src="glyphs/small.svg" width="40"><br><sub>small</sub></td>
<td align="center"><img src="glyphs/same.svg" width="40"><br><sub>same</sub></td>
<td align="center"><img src="glyphs/BE.svg" width="40"><br><sub>BE</sub></td>
<td align="center"><img src="glyphs/wait.svg" width="40"><br><sub>wait</sub></td>
</tr>
</table>

---

## How it works

### 1. Semantic Tokenizer

Any English text is compressed into 88 concepts. Each word maps to the nearest hieroglyph through a 2000+ word synonym map with morphological fallbacks:

```
"the old dog stretched by the fireplace and fell asleep"
  → before animal fire and sleep

"she started a new company and worked on it day and night"
  → woman other work dark
```

### 2. Transformer

A GPT-class transformer learns patterns in the compressed glyph space. Two training modes:

- **Diffusion** (recommended) — randomly masks positions, trains bidirectional prediction. At inference, starts from all-MASK and iteratively reveals glyphs by confidence. The cave painting appears all at once.
- **Autoregressive** — standard left-to-right next-token prediction.

### 3. Hebbian Plasticity

The cave learns from every conversation. Low-rank Hebbian LoRA adapters on Q and V projections update after each generation — no backprop, no tape, pure online co-occurrence. Neurons that fire together wire together. This is the fast, always-on layer of adaptation. For deeper consolidation see [§8 — Mass-Threshold CPT](#8-mass-threshold-continued-pre-training).

### 4. Symbol Emergence + Natural Selection

Birth is free — survival is not. When two glyphs co-occur strongly (>0.75), a new combined symbol is born. It survives as long as the parent pair keeps co-occurring (current strength ≥ 0.525, i.e. 0.7 × the birth threshold); when the pattern fades, the symbol dies. Any actual usage during generation also extends life. Depth cap: 5 levels, then freeze as a new primitive (like "breakfast" lost "break+fast").

We fed it Dracula via `feed/` — 7029 sentences devoured (SPA .!? split). Twelve symbols were born; nine died when their parent co-occurrence decayed below 0.525; three are still alive: `and+me` (0.999), `and+BE` (0.778), `me+BE` (0.715) — exactly the patterns a first-person gothic novel would reinforce. Evolution works.

### 5. Async Self-Learning (SPA Sentence Phonons)

A background thread watches the `feed/` directory. Drop any `.txt` file there — the model splits it into sentences (phonons, per [SPA from Q](https://github.com/ariannamethod/q)), runs each through the semantic tokenizer, and updates Hebbian weights autonomously. Passive reading = 0.3x signal, V-only. The cave reads while you sleep.

```
cp data/dracula.txt feed/
  [learner] consuming dracula.txt (890K)...
  [learner] dracula.txt → 7029 sentences learned
```

### 6. BE — The Super-Verb

One circle. **BE** turns any noun into a verb: `BE fear` = to be afraid. `BE love` = to love. `BE fire` = to burn. One symbol that doubles the expressiveness of the entire language.

### 7. The Ring — two caves talking

`./cavellman` runs two engines, **A** (extrovert, coherence_floor 0.30) and **B** (introvert, 0.60), on top of a single `CaveField` apiece: an **excitement** accumulator (surprise × novelty), a **dissonance** meter (for unexpected pairs), and a drifting silence threshold à la [Stanley](https://github.com/ariannamethod/stanley)'s silence-gate. An engine speaks only when its excitement trips the floor — or when dissonance > 0.40 forces a tunneled outburst, AML-style. Otherwise it stays silent, and the silence itself is data. **Maturity drift** (±0.005 per turn, clamped ±0.30) auto-calibrates the floor: if an engine hogs the ring (> 70% speaking), its gate tightens; if it barely speaks (< 20%), it loosens.

The ring does not need you. Engines talk to each other; the feed/ folder feeds them; you can drop into the ring by typing glyphs, but you are one more source among equals — not the center. Sometimes neither engine answers. Sometimes both do. That's the field, not a chatbot.

Shipped weights include two distinct voices — `cavellman_A.bin` trained on Dracula only (15K steps, seed 42, first-person `me`-dominated) and `cavellman_B.bin` trained on Frankenstein only (15K steps, seed 123, formal epistolary). Default run uses both:

```bash
./cavellman                                     # A=Dracula, B=Frankenstein
./cavellman --weights weights/cavellman_medium.bin --preset medium   # shared medium weights
```

```
[B] BE not
[A] one BE and
[B] me man have
[A] me and
[A] after me BE
[B] and me
[user] love woman child
[B] woman and
[A] good and me have woman BE woman me have
[B] me
[B] have me and me
[A] BE see
```

Seven symbols emerged in 59 ticks (`not+BE`, `cold+man`, `man+me`, `and+me`, `me+BE`, `me+have`). Maturity drifted: A 0.30 → 0.20, B 0.60 → 0.50 — both loosened their gates because the ring stayed sparse.

### 8. Mass-Threshold Continued Pre-Training

Hebbian adapters react fast but don't reshape the underlying embeddings. For deeper consolidation each engine in the dual ring carries a Arianna-style mass accumulator with three counters:

- **bytes** — raw volume of heard/spoken glyph text captured in the engine's holding buffer (`feed/<name>_holding.txt`)
- **novelty** — cumulative surprise × novelty signal per token (same quantity that drives excitement)
- **resonance** — cumulative excitement integral across ticks

When all three trip their thresholds (currently `2500 bytes + novelty ≥ 8 + resonance ≥ 15`), the engine forks `train_cavellman --start-from <current.bin>` on its own holding buffer. The child does ~300 steps of proper notorch CPT — tape, backward, Chuck optimizer, full-param updates — while the dual dialogue keeps running in the parent. When the child finishes, the parent atomically memcpys the new tensor data back into the live `CaveModel` (KV cache, emerged symbols, Hebbian adapters all stay intact — only the raw weights swap).

Current thresholds: **2500 bytes + novelty ≥ 8 + resonance ≥ 15** per engine, **300 CPT steps** per burst.

From a real `--preset medium --weights cavellman_medium.bin` session driven by ~20 mixed user prompts (nature / emotion / mind / BE-verbs / unexpected combos like `AI dream death music`) over ~10 minutes. After 896 ticks B tripped first:

```
[B] microtrain spawned (pid=29925, 2601 bytes / nov 425.9 / res 575.8)
...child notorch CPT runs in parallel, parent loop keeps talking...
[B] microtrain #1 done — 36 tensors swapped (weights live)
```

Snippets from the same session — emerged composites actually surfacing in output:

```
[A] man+me me and body       ← composite used in generation
[B] BE woman BE               ← BE super-verb chain
[A] know me and+me            ← "know" picked up mid-session
[B] one and+BE me             ← another composite in speech
```

Maturity drifted both gates to their lower clamps: A 0.30 → 0.00, B 0.60 → 0.30. Speak ratio settled at ~14% — sparse dialogue with bursty user input is the equilibrium.

So the cave runs on two learning clocks: Hebbian on every turn (fast, shallow), CPT on accumulated mass (slow, deep). Whatever it has been hearing — the other engine, the user, or a book dropped into `feed/` — becomes new weights.

Safeguards (planned, not yet in code): sha256 whitelist of approved sources, holding-area gate with `!learn <hash>` command, unknown-word ratio rejection. For now the holding buffer is trusted — don't expose it to untrusted text.

---

## Quick Start

### cavellman.c — the ring

```bash
make cavellman                     # build with BLAS + pthreads

./cavellman                                                            # A=Dracula, B=Frankenstein (default)
./cavellman --preset medium --weights weights/cavellman_medium.bin     # shared medium weights
./cavellman --weights-a weights/cavellman_A.bin \
            --weights-b weights/cavellman_B.bin                        # explicit A/B
```

```
══════════════════════════════════════════════════════════
  caveLLMan — DUAL MODE
══════════════════════════════════════════════════════════
  A: extrovert (coherence_floor 0.30)
  B: introvert (coherence_floor 0.60)
  tunnel=0.40, decay=0.94/0.90, maturity drift ±0.30
──────────────────────────────────────────────────────────

[B] me BE
[A] and one me
[user] love woman child
[B] woman and
[A] good and me have woman BE
  *** SYMBOL EMERGED: me+BE (id=88, depth=1, strength=0.999) ***
[B] and+BE me
```

Drop `.txt` files into `feed/` — Dracula, Frankenstein, anything. Both caves devour them sentence by sentence.

### Training (notorch C)

```bash
make train                         # build train_cavellman + train_diffusion
# assemble a corpus of raw English
cat data/dracula.txt data/frankenstein.txt data/miller.txt \
    data/dante.txt data/suppertime.txt data/fineweb_edu.txt > data/corpus_big.txt

./train_cavellman --dataset data/corpus_big.txt --preset medium --steps 20000
./train_diffusion  --dataset data/corpus_big.txt --steps 15000

# continued pre-training from an existing checkpoint
./train_cavellman --start-from weights/cavellman_medium.bin \
                  --dataset data/my_new_text.txt --preset medium --steps 500
```

`train_cavellman` reads raw English, splits on `.!?` (SPA phonons), and compresses each sentence through the same `semantic_tokenizer.h` the engine uses at inference — train and runtime share one source of truth for the 88 canonical glyphs. `--start-from FILE` is what the ring's mass-threshold CPT uses under the hood.

---

## Architecture

```
                    ┌──────────────┐
                    │  feed/*.txt  │ ← drop any text here
                    └──────┬───────┘
                           │ async thread
┌─────────────────┐        ▼
│  Any English    │──▶ Semantic Tokenizer ──▶ 88 Glyph IDs
│  text           │    2060 words → 88       (fixed vocab)
└─────────────────┘                                │
                                                   ▼
                              ┌──────────────────────────────┐
                              │  Transformer + Hebbian LoRA  │
                              │  rank=4 on Q,V projections   │
                              │  prediction error signal     │
                              ├──────────────────────────────┤
                              │  Co-occurrence → emergence   │
                              │  Birth free, survive while   │
                              │    parent co-occ ≥ 0.525     │
                              │  Depth cap 5 → freeze        │
                              └──────────────────────────────┘
                                           │
         ┌─────────────────────────────────┼────────────────────┐
         ▼                                 ▼                    ▼
┌──────────────────┐           ┌──────────────────────┐  ┌───────────────┐
│  Dual CaveField  │           │  Mass-threshold CPT  │  │ SVG output    │
│  excitement      │           │  bytes+nov+resonance │  │ 88 base +     │
│  coherence_floor │──triggers▶│  → fork train_cave-  │  │ emerged signs │
│  dissonance      │           │  llman --start-from  │  └───────────────┘
│  maturity drift  │           │  → atomic reload     │
└──────────────────┘           └──────────────────────┘
```

## Numbers

| Metric | Value |
|--------|-------|
| Base alphabet | 88 hieroglyphs |
| Max emerged | 128 new symbols |
| Semantic map | 2060 English words |
| Hebbian rank | 4 (LoRA on Q, V) |
| Hebbian signal | prediction error [0.1, 2.0] |
| Emergence threshold | 0.75 co-occurrence |
| Survival | parent co-occ ≥ 0.525 (or ≥ 5 uses) within 500 interactions |
| Depth cap | 5 levels, then freeze as primitive |
| Sentence splitter | SPA phonons (.!?) |
| C model (small)  | 472K params (1.89 MB) |
| C model (medium) | 826K params (3.23 MB) |
| Browser model | ~31K params |
| Dual mode field | excitement + dissonance + drifting coherence_floor |
| Silence gate | speak iff `excitement > floor` or `dissonance > 0.40` |
| Maturity drift | ±0.005 / turn, clamped ±0.30 around baseline |
| CPT trigger | 2500 bytes + 8 novelty + 15 resonance (per engine) |
| CPT burst | 300 notorch steps, Chuck optimizer, full-param updates |
| Engine | [notorch](https://github.com/ariannamethod/notorch) (pure C, BLAS) |
| State file | `weights/cavellman.state` |
| Shipped weights | v3 (mixed), A (Dracula), B (Frankenstein), medium (12.7 MB corpus) |

## License

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later). See [`LICENSE`](LICENSE).

## Credits

88-glyph alphabet inspired by Genevieve von Petzinger's 32 cave signs. Originally forked from [emojiGPT](https://github.com/MattWenJun/emojiGPT) by @MattWenJun (who forked Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)). Rebuilt from scratch: semantic tokenizer, Hebbian plasticity, symbol emergence with natural selection, SPA sentence phonons, async self-learning, diffusion engine, cave-painting SVG hieroglyphs, C engine on [notorch](https://github.com/ariannamethod/notorch). Dual-mode silence-gate physics borrowed from [Stanley](https://github.com/ariannamethod/stanley), tunnel/resonance primitives from [AML](https://github.com/ariannamethod/ariannamethod.ai), mass-threshold CPT pattern from [arianna.c](https://github.com/ariannamethod/arianna.c). Corpora include Gutenberg public-domain texts, a FineWeb-EDU sample, and Oleg Ataev's own *SUPPERTIME v2.0*. — [Arianna Method](https://github.com/ariannamethod).

---

## Appendix A — a cave ring transcript

Default `./cavellman` loads `cavellman_A.bin` (Dracula, extrovert) and `cavellman_B.bin` (Frankenstein, introvert), wires a shared `feed/` learner so both caves hear anything dropped in, and opens the ring. A human may type glyphs into the ring, or stay silent. The caves do not need you.

```
$ ./cavellman
  loaded 36/36 tensors, Hebbian rank=4
  loaded 36/36 tensors, Hebbian rank=4
══════════════════════════════════════════════════════════
  caveLLMan — DUAL MODE
══════════════════════════════════════════════════════════
  A: extrovert (coherence_floor 0.30)
  B: introvert (coherence_floor 0.60)
  tunnel=0.40, decay=0.94/0.90, maturity drift ±0.30
  type glyphs any time to join the ring. 'quit' to exit.
  [learner] watching feed/ for .txt files — both caves hear.
──────────────────────────────────────────────────────────

> stats
  [A] exc=0.02 floor=0.30 diss=0.00 spoke=0/39
  [B] exc=0.02 floor=0.60 diss=0.00 spoke=0/39

> dark fear fire
[user] dark fear fire
  *** SYMBOL EMERGED: fire+fear    (id=88, depth=1, strength=0.999) ***
[A] and BE
  *** SYMBOL EMERGED: and+BE       (id=88, depth=1, strength=0.999) ***
[B] one
  *** SYMBOL EMERGED: dark+fear    (id=89, depth=1, strength=0.999) ***
[A] and woman and dark+fear
  *** SYMBOL EMERGED: woman+and    (id=89, depth=1, strength=0.999) ***
[B] me have
  *** SYMBOL EMERGED: me+have      (id=90, depth=1, strength=0.999) ***
[A] me
```

Notes on what actually happens here:

- `./cavellman` takes no flags. The single-engine dialogue loop has been deprecated — a human-in-center chatbot is not what this repo is for anymore.
- Five composite symbols are born from a three-glyph user prompt: `fire+fear`, `and+BE`, `dark+fear`, `woman+and`, `me+have`. Their numeric ids (88+) share the transformer head's output slots and can therefore be sampled back into the ring.
- **A actually uses a composite mid-sentence:** the fourth line `and woman and dark+fear` contains the id-89 symbol that was born one tick earlier. The ring's own emergence is feeding back into its generation on the same interaction.
- A and B have asymmetric silence gates (0.30 vs 0.60), so they disagree on when to speak. Sometimes only one answers; sometimes neither. Silence is a legal response.
- The `[learner] watching feed/` line is not decoration — drop any `.txt` into `feed/` while this is running and both caves will devour it in passive-reading mode (0.3× signal, V-only). The engines' own `*_holding.txt` CPT buffers are skipped.
