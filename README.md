# caveLLMan

### 32 hieroglyphs. universal language. 30,000 years in the making.

No Python. No pip. No torch. Pure C.

---

## The 32 Signs

In 2016, paleoanthropologist Genevieve von Petzinger published research documenting exactly **32 recurring geometric signs** found across 146 European cave sites spanning 30,000 years. Lines, chevrons, circles, spirals, zigzags, hands — the same symbols appearing from France to Indonesia to Australia. Not random. Not decoration. A shared cognitive vocabulary predating all known writing systems by 25,000 years.

caveLLMan is a GPT that speaks this language. 32 abstract hieroglyphs — semantic atoms that encode all of human experience. Not emoji. Not words. Signs. The kind a cave painter would recognize.

The model learns which atoms follow which. Light follows dark. Grief follows love. Creation follows destruction. The grammar of being alive, compressed into 32 symbols and a tiny transformer.

## Quick Start — C Engine (production)

```bash
# Build (no dependencies, portable)
make cpu

# Train
./train_emolm --preset standard --steps 4000
# Saves to weights/emolm.bin by default

# Infer (interactive emoji completion)
./infer_emolm --weights weights/emolm.bin --preset standard

# Or specific save path
./train_emolm --preset tiny --steps 2000 --save weights/tiny.bin
./infer_emolm --weights weights/tiny.bin --preset tiny
```

Built on [notorch](https://github.com/ariannamethod/notorch) — PyTorch replacement in pure C. Tensors, autograd, optimizers, no dependencies.

### Benchmark

| Engine | Preset | Steps | Time | Speed |
|--------|--------|-------|------|-------|
| **C (notorch)** | tiny | 2000 | ~1.4s | 1452 steps/s |
| **C (notorch)** | standard | 4000 | ~45s | 89 steps/s |
| **C (notorch)** | medium | 8000 | ~12min | 11 steps/s |
| Python (emolm.py) | tiny | 2000 | ~34min | 1 step/s |

C is **~1400x faster** than Python at the same architecture.

### Presets

| Preset | Params | Embd | Heads | Layers | CTX | Default steps |
|--------|--------|------|-------|--------|-----|---------------|
| `tiny` | ~16K | 18 | 3 | 2 | 32 | 2000 |
| `micro` | ~100K | 48 | 4 | 3 | 64 | 3000 |
| `standard` | ~200K | 64 | 8 | 3 | 64 | 4000 |
| `small` | ~500K | 96 | 8 | 4 | 128 | 5000 |
| `medium` | ~1M | 128 | 8 | 4 | 128 | 8000 |

### Sampling

Generation uses **top-p (nucleus) sampling** — only samples from the smallest set of tokens whose cumulative probability exceeds p=0.9. Combined with temperature scaling (default 0.8), produces coherent stories without degenerate repetition.

## Browser (index.html)

1. Open `index.html` — training starts automatically
2. Watch the loss curve drop in real time
3. Use the emoji keyboard to compose a prompt (e.g. `🌅 ☕ 🏃`)
4. Press **Continue ▶** and watch the model predict the story

WebGPU detected? GPU-accelerated. Not available? Silent fallback to CPU.

- `index.html` — Hebrew RTL (root language)
- `en/index.html` — English

## Python (educational/reference)

```bash
python emolm.py                              # train + interactive mode
python emolm.py --preset tiny                # ~16K params
python emolm.py --preset standard            # ~200K params
python emolm.py --optimizer chuck            # Chuck optimizer
python emolm.py --prompt "🌅 ☕ 🏃"          # single prompt
```

Zero dependencies — pure Python standard library. Educational reference implementation.

## The 32 Hieroglyphs

```
ELEMENTS    light    dark     water    fire
EARTH       tree     mountain home     path
BODY        strength pain     sleep    food
EMOTION     joy      grief    love     fear     anger
PEOPLE      person   child    group
MIND        idea     knowledge sound   prayer
ACTION      create   destroy  seek     death
GRAMMAR     not      many     when     and
```

32 symbols. Each one abstract — a semantic atom, not a picture. Combinations create meaning:

```
light + water  = morning by the sea
grief + child  = loss of innocence
fire  + anger  = rage
path  + seek   = pilgrimage
create + knowledge = discovery
person + not   = loneliness
```

Custom SVG glyphs in `glyphs/` — cave-painting aesthetic, minimal strokes, works at any size.

## Architecture

```
Token Embedding + Positional Embedding
          ↓
       RMSNorm
          ↓
  ┌─── Transformer Block (×N) ───┐
  │  RMSNorm → Multi-Head Attn   │
  │  + Residual                   │
  │  RMSNorm → FFN (SiLU)        │
  │  + Residual                   │
  └───────────────────────────────┘
          ↓
     Linear → Softmax → Loss
```

### Optimizers

| Optimizer | Formula |
|-----------|---------|
| `adam` | θ -= α × m̂/(√v̂ + ε) |
| `chuck` | θ -= (α × S × λ) × m̂/(√v̂ + ε) + η |

[Chuck](https://github.com/ariannamethod/chuck.optimizer): self-aware optimizer — global λ dampen, stagnation noise η, macro patience S.

## Datasets

### Hebrew (data/) — root language

| Dataset | Stories | Description |
|---------|---------|-------------|
| emoji_stories | 163 | Israeli-themed emoji stories |
| hebrew_stories | 92 | Colloquial Hebrew prose |
| exhale_he | 2500 | Klaus somatic reaction corpus |
| paired_he | 33 | emoji\|\|\|Hebrew pairs |

### English (en/data/)

| Dataset | Stories | Description |
|---------|---------|-------------|
| emoji_core | 40 | 32-emoji alphabet (default) |
| emoji_stories | 231 | Full 8-emoji stories |
| exhale_en | 2507 | Klaus somatic reaction corpus |
| paired_en | bidir | emoji\|\|\|text pairs |

## Tests

```bash
# All tests (C + Python)
make test

# C only (smoke test)
make test_c

# Python only (131 tests)
make test_py
# or
python -m unittest tests.test_emolm -v

# Diffusion tests (12 tests)
python -m unittest tests.test_diffusion -v
```

## Project Structure

```
emoLM/
├── train_emolm.c    # C training engine (production)
├── infer_emolm.c    # C inference (interactive generation)
├── notorch.c/.h     # notorch — pure C tensor autograd
├── Makefile          # make cpu / make train / make infer / make test
├── emolm.py          # Python reference implementation
├── diffusion.py      # Discrete emoji diffusion engine
├── index.html        # Browser engine — Hebrew RTL (auto-trains)
├── en/index.html     # Browser engine — English
├── data/             # Hebrew datasets (root)
├── en/data/          # English datasets
├── weights/          # Saved weights (.bin + .json metadata + .vocab)
├── icons/            # 32 hieroglyphic SVG pictograms (1:1 with emoji alphabet)
├── tests/            # Python tests (131 + 12 diffusion)
└── .github/workflows/ci.yml  # CI: C build + smoke test + Python tests
```

## Weights Format

Training auto-saves to `weights/emolm.bin` with sidecar files:
- `emolm.bin` — binary weights (notorch nt_save format)
- `emolm.bin.json` — metadata (preset, steps, loss, config)
- `emolm.bin.vocab` — vocabulary (one emoji per line)

Use `--no-save` to skip saving.

## Why 32 emoji?

Emoji are dimensionality reduction. A single 🏥 carries what takes a full sentence in English. By limiting to 32 symbols — each visually distinct, semantically unambiguous — the model learns faster and the output is immediately readable.

The other half is you. Your brain unpacks each emoji into a rich scene. The model writes compressed stories; your mind decompresses them.

## Ecosystem

Part of the [Arianna Method](https://github.com/ariannamethod) family:

| Project | Description |
|---------|-------------|
| [**Klaus**](https://github.com/ariannamethod/klaus.c) | Somatic emotion engine — 6 coupled oscillators, exhale corpora |
| [**q**](https://github.com/ariannamethod/q) | Resonant reasoning engine — triple attention, RRPRAM |
| [**postgpt**](https://github.com/ariannamethod/postgpt) | Ghost model — metaweights, zero-training inference |
| [**notorch**](https://github.com/ariannamethod/notorch) | PyTorch replacement in pure C — tensors, autograd, optimizers |
| [**chuck.optimizer**](https://github.com/ariannamethod/chuck.optimizer) | Self-aware optimizer — 9 levels, competes with Adam |

## Language Agnostic

Emoji have no language barrier. The 32-symbol alphabet is universal — a reader in Tokyo and a reader in São Paulo both understand 🤕→🏥→💉→🙏→😊 as the same story. The model doesn't care what language your datasets are in — swap `emoji_core.txt` for any language's paired data and it learns the same patterns.

Currently ships with:
- **Hebrew** (root) — `data/` — RTL index.html
- **English** — `en/data/` — LTR en/index.html

Adding a new language = adding a `paired_XX.txt` + stories file. No code changes.

## Credits

Forked from the original [**emojiGPT**](https://github.com/MattWenJun/emojiGPT) by [**@MattWenJun**](https://github.com/MattWenJun) — a single-file browser GPT built on [**@karpathy**](https://github.com/karpathy)'s [microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) gist.

What we added: C production engine ([notorch](https://github.com/ariannamethod/notorch)), Hebrew as root language, 2500+ somatic datasets from [Klaus](https://github.com/ariannamethod/klaus.c), [Chuck optimizer](https://github.com/ariannamethod/chuck.optimizer), five model presets (16K→1M params), bidirectional training, discrete diffusion, and SVG hieroglyphic icons — by [Arianna Method](https://github.com/ariannamethod).

