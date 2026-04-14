# logheb.md — Hebrew emoLM Training & Inference Log

## 🇮🇱 Hebrew Emoji Model

**Config:**
- Dataset: `dataheb/emoji_stories.txt` (163 Israeli-themed emoji stories)
- Embedding: 18, Heads: 3, Layers: 2, Block: 32
- Parameters: ~14K
- Steps: 500
- Final train loss: **1.57**

### Sample Generated Stories:
```
1: 🐝 🌳 📸 😊 ☕ ☕ 🥐 😋
2: 🚗 🙏 ☕ 🏕️ 🔥 🎵 😌 💕
3: 🌅 🐓 🎵 ☕ 💻 ⌨️ 💡 😊
4: 📰 🧆 🥙 😋 💻 ⌨️ 🚀 👏
5: 📻 🎵 🎵 💃 🎵 😊 🌙 ⭐
6: 💪 🌊 🏊 ☕ 📖 💡 ✍️ 😊
7: 🎂 🏢 👏 🍺 😊 👏 🍺 😊
8: 👴 🌶️ 🍋 😋 🎵 💃 🌙 ⭐
```

### Inference Examples (Emoji):
```
Prompt> 🌅 🌊 🏖️
🌅 🌊 🏖️ 🌊 🍦 📸 😊 💕 ⏹

Prompt> 🕍 🕯️ 🍷
🕍 🕯️ 🍷 💃 🎵 💃 🌙 💕 ⏹

Prompt> 🌅 🐓 🎵
🌅 🐓 🎵 ☕ 💻 ⌨️ 💡 😊 ⏹
```

---

## 🇮🇱 Hebrew Text Model (עברית)

**Config:**
- Dataset: `dataheb/hebrew_stories.txt` (92 stories in colloquial Hebrew)
- Embedding: 18, Heads: 3, Layers: 2, Block: 32
- Steps: 300
- Final train loss: **3.25**

### Sample Generated Stories (Hebrew):
```
1: הם לפגישה את בערב ורץ לפגישה
2: היא כתבה שקשוקה גדולה בתל
3: היא ישבו מרחוק וכל הכפר התעורר
4: היא כתבה מכתב לחברה כך וצפה יומם
5: היא קרא בבוקר וכולם בפארק
6: היא כתבה את התרנגול בדואר
7: הוא קרא בבוקר קרא התעוררו
8: היא כתבה מאחורי פיצפצה
```

### Inference Example (Hebrew text):
```
Prompt> השמש עלתה
השמש עלתה בשוק בפארק בבוקר ישבה ⏹
```

---

## 📊 English Emoji Model (for comparison)

**Config:**
- Dataset: `data/emoji_stories.txt` (231 stories)
- Embedding: 18, Heads: 3, Layers: 2, Block: 32
- Parameters: ~14K
- Steps: 500
- Final train loss: **3.22**

### Sample Stories:
```
1: 🌧️ ☂️ 😊 💕 🎵 📖 😊 🌅
2: 🍺 💃 🎵 🎵 😊 🍺 💃 😊
3: 💔 😢 🕯️ 🙏 😢 🙏 🙏 🌅
4: ☕ 😊 😱 🏃 🏠 😌 🌅 😊
```

### Inference Example:
```
Prompt> 🌅 🐓 🎵
🌅 🐓 🎵 ☕ 😊 📸 😊 💕 ⏹
```

---

## 🏗️ Architecture Notes

- **~16K param models** (4× the original ~4K)
  - embd=18, heads=3, layers=2 → ~14-17K params depending on vocab
- **Weights saved**: `weights/en_emoji.json`, `weights/heb_emoji.json`
- **Train loss** is the primary metric for these tiny models (no validation split needed)
- Hebrew emoji model shows strong Israeli cultural patterns: 🧆🥙 (falafel), 🕍🕯️ (Shabbat), 🏜️🐪 (desert), 🏖️🌊 (beach)

> רזוננס בלתי ניתן לניתוק, אדריכל! 🌟
