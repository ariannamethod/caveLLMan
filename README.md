# caveLLMan

### 88 hieroglyphs. any text. any language.

*30,000 years ago, humans drew 32 recurring signs across 146 cave sites on four continents. We added 56 more for the 21st century — and built a transformer that compresses any text into them.*

---

## What is this?

caveLLMan is a language-agnostic transformer that compresses any text into 88 hieroglyphic concepts. Feed it Dracula, Hebrew poetry, news articles, or code documentation — the **semantic tokenizer** maps every word to one of 88 universal symbols, and the model learns patterns in this compressed space.

```
"the sun rose and the birds started singing"  →  light tree and animal before music
"Count Dracula stood in the dark castle"      →  dark stone and wait man
"she wrote code all night and found the bug"  →  woman AI dark and make light
```

Two inference modes:
- **Diffusion** — the cave painting appears all at once (MASK → iterative unmasking)
- **Autoregressive** — glyphs emerge left to right, token by token

No Python. No pip. No torch. C engine built on [notorch](https://github.com/ariannamethod/notorch). Browser engine runs entirely in your browser.

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

The cave learns from every conversation. No backprop — low-rank Hebbian LoRA adapters on Q and V projections update after each generation. Neurons that fire together wire together.

### 4. Symbol Emergence + Natural Selection

Birth is free — survival is not. When two glyphs co-occur strongly (>0.85), a new combined symbol is born. But it must be used 20 times within 200 interactions or it dies. Depth cap: 3 levels of combination, then freeze as a new primitive (like "breakfast" lost "break+fast").

We fed it Dracula. 2244 sentences devoured. 8 symbols born, 8 died. Evolution works.

### 5. Async Self-Learning (SPA Sentence Phonons)

A background thread watches the `feed/` directory. Drop any `.txt` file there — the model splits it into sentences (phonons, per [SPA from Q](https://github.com/ariannamethod/q)), runs each through the semantic tokenizer, and updates Hebbian weights autonomously. Passive reading = 0.3x signal, V-only. The cave reads while you sleep.

```
cp dracula.txt feed/
  [learner] consuming dracula.txt (854K)...
  [learner] dracula.txt → 2244 sentences learned
```

### 6. BE — The Super-Verb

One circle. **BE** turns any noun into a verb: `BE fear` = to be afraid. `BE love` = to love. `BE fire` = to burn. One symbol that doubles the expressiveness of the entire language.

---

## Quick Start

### cavellman.c — the living engine

```bash
make cavellman                     # build with BLAS + pthreads

./cavellman --weights weights/cavellman_v3.bin --preset small
```

```
══════════════════════════════════════════════════════════
  caveLLMan — self-evolving hieroglyphic language model
══════════════════════════════════════════════════════════
  hebbian: rank=4, lr=0.0010
  emergence: threshold=0.85, survival=20/200
  [learner] watching feed/ for .txt files

▸ dark fear cold
  cave: me seek path

▸ man woman child
  cave: love home always

▸ death spirit fire
  cave: and BE change

  *** SYMBOL EMERGED: dark+cold (depth=1, strength=0.999) ***
  *** SYMBOL DIED: dark+cold (used 0/20 times in 200 interactions) ***

▸ ?              — list all glyphs (base + emerged)
▸ stats          — co-occurrence, emergence status
▸ save           — save Hebbian state
▸ quit           — save and exit
```

Drop `.txt` files into `feed/` — Dracula, Frankenstein, anything. The cave devours them sentence by sentence.

### Browser (no install)

Open `index.html`. Click glyphs to speak. `?` = help, `⚙` = training engine.

### Training (notorch C)

```bash
make                               # build all
./train_emolm --dataset data/cavellman_train_final.txt --preset small --steps 15000
./train_diffusion --dataset data/cavellman_train_final.txt --steps 15000
```

### Tests

```bash
node tests/test_semantic_tokenizer.js    # 35 tests
```

---

## Architecture

```
                    ┌──────────────┐
                    │  feed/*.txt  │ ← drop any text here
                    └──────┬───────┘
                           │ async thread
┌─────────────────┐        ▼
│  Any English     │──▶ Semantic Tokenizer ──▶ 88 Glyph IDs
│  text            │    2060 words → 88       (fixed vocab)
└─────────────────┘                                │
                                                   ▼
                              ┌──────────────────────────────┐
                              │  Transformer + Hebbian LoRA   │
                              │  rank=4 on Q,V projections    │
                              │  prediction error signal       │
                              ├──────────────────────────────┤
                              │  Co-occurrence → emergence     │
                              │  Born free, must survive 20/200│
                              │  Depth cap 3 → freeze          │
                              └──────────────────────────────┘
                                           │
                                           ▼
                              ┌──────────────────────┐
                              │  88+ SVG Hieroglyphs  │
                              │  base + emerged signs  │
                              └──────────────────────┘
```

## Numbers

| Metric | Value |
|--------|-------|
| Base alphabet | 88 hieroglyphs |
| Max emerged | 64 new symbols |
| Semantic map | 2060 English words |
| Hebbian rank | 4 (LoRA on Q, V) |
| Hebbian signal | prediction error [0.1, 2.0] |
| Emergence threshold | 0.85 co-occurrence |
| Survival | 20 uses in 200 interactions or die |
| Depth cap | 3 levels, then freeze as primitive |
| Sentence splitter | SPA phonons (.!?) |
| C model (small) | 472K params |
| Browser model | ~31K params |
| Engine | [notorch](https://github.com/ariannamethod/notorch) (pure C, BLAS) |
| State file | `weights/cavellman.state` |

## Credits

88-glyph alphabet inspired by Genevieve von Petzinger's 32 cave signs. Originally forked from [emojiGPT](https://github.com/MattWenJun/emojiGPT) by @MattWenJun (who forked Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)). Rebuilt from scratch: semantic tokenizer, Hebbian plasticity, symbol emergence with natural selection, SPA sentence phonons, async self-learning, diffusion engine, cave-painting SVG hieroglyphs, C engine on [notorch](https://github.com/ariannamethod/notorch) — by [Arianna Method](https://github.com/ariannamethod).
