# caveLLMan

### 88 hieroglyphs. one super-verb. the grammar of being alive.

*30,000 years ago, humans drew 32 recurring signs across 146 cave sites on four continents. We added 56 more for the 21st century — and taught a transformer to speak them.*

---

## What is this?

caveLLMan is a GPT that communicates through abstract hieroglyphs — not emoji, not words, but semantic atoms. Each symbol is a concept so fundamental that a cave painter and a software engineer would both recognize it.

The model learns which atoms follow which. Light follows dark. Grief follows love. Creation follows destruction. **BE** turns any noun into a verb: `BE fear` = to fear. `BE love` = to love. `BE fire` = to burn. One symbol that doubles the expressiveness of the entire language.

```
me BE change and choose good now
never lie and always speak and know
old woman remember and give child know
other man speak and me not agree
```

No Python. No pip. No torch. Pure C, built on [notorch](https://github.com/ariannamethod/notorch).

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

## How to read caveLLMan

Each story is a sequence of glyphs. You read them like a sentence:

```
me BE fear dark not see path
↓   ↓   ↓    ↓    ↓   ↓   ↓
I  am  afraid dark can't see the way
```

```
woman give child food and love
↓      ↓    ↓     ↓    ↓   ↓
a woman gives a child food and love
```

```
old man remember before and BE grief
↓   ↓      ↓       ↓     ↓  ↓   ↓
an old man remembers the past and grieves
```

## BE — The Super-Verb

One circle. The most powerful symbol in the alphabet. **BE** turns any noun into a verb, any concept into a lived state:

| caveLLMan | meaning |
|-----------|---------|
| me **BE** fear | I am afraid |
| woman **BE** love man | a woman loves a man |
| child **BE** joy | the child is happy |
| spirit **BE** light | the spirit is luminous |
| AI **BE** know much | the AI knows a lot |
| me **BE** free | I am free |
| person **BE** death | a person dies |
| fire **BE** change | fire transforms |

Without BE, these are nouns: fear, love, joy. With BE, they become experience. A cave painter 30,000 years ago would draw a circle around a figure to mean "this is happening." We kept the tradition.

---

## Quick Start

```bash
make                    # build with BLAS acceleration
make cpu                # build without BLAS (portable)

./train_emolm --dataset data/cavellman_train_final.txt --preset small --steps 15000
./infer_emolm --weights weights/cavellman_v3.bin --preset small
```

## What the model generates

```
me BE change and choose good now        — I am changing and choosing good now
child question and know                 — a child asks and learns
other woman help and give food child    — another woman helps and feeds a child
never lie and always speak and know     — never lie, always speak and know
me BE anger cause you lie               — I am angry because you lied
spirit BE light not body not pain       — spirit is light, no body, no pain
man woman bond love home child          — man and woman bond in love, home, children
```

## Numbers

| Metric | Value |
|--------|-------|
| Alphabet | 88 hieroglyphs |
| Training data | 15,500 stories |
| Model (small) | ~500K params |
| Training loss | 4.49 → 1.66 |
| Training speed | 20 steps/s on Mac |
| Engine | notorch (pure C, BLAS) |

## Why 88?

Genevieve von Petzinger found 32 recurring geometric signs in European cave art spanning 30,000 years. That's enough to be human.

We added 56 more to be a human in the 21st century: internet, AI, money, stress, free, work, choose. Plus a super-verb that the cave painters already knew — BE.

88 symbols. Any story. Any language. Any century.

## Credits

Originally inspired by [emojiGPT](https://github.com/MattWenJun/emojiGPT) by @MattWenJun. Rebuilt from scratch: 88-glyph alphabet designed with 6 AI linguists, cave-painting SVG hieroglyphs, C engine on [notorch](https://github.com/ariannamethod/notorch), BE super-verb — by [Arianna Method](https://github.com/ariannamethod).
