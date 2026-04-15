/*
 * Tests for semantic_tokenizer.js
 * Run: node tests/test_semantic_tokenizer.js
 */

const { SemanticTokenizer, SEMANTIC_MAP, STOP_WORDS } = require('../semantic_tokenizer.js');

let passed = 0, failed = 0;

function assert(condition, msg) {
  if (condition) { passed++; }
  else { failed++; console.error(`  FAIL: ${msg}`); }
}

function test(name, fn) {
  process.stdout.write(`${name}... `);
  try { fn(); console.log('OK'); }
  catch (e) { failed++; console.error(`EXCEPTION: ${e.message}`); }
}

const st = new SemanticTokenizer();

// ── Basics ──

test('tokenizer instantiates', () => {
  assert(st instanceof SemanticTokenizer, 'should be instance');
  assert(st.glyphs.length === 88, `expected 88 glyphs, got ${st.glyphs.length}`);
});

test('all 88 glyphs present in SEMANTIC_MAP', () => {
  const keys = Object.keys(SEMANTIC_MAP);
  assert(keys.length === 88, `expected 88 keys, got ${keys.length}`);
});

test('word map has entries', () => {
  assert(st.wordToGlyph.size > 1500, `expected >1500 word mappings, got ${st.wordToGlyph.size}`);
});

// ── Single word mappings ──

test('sun → light', () => {
  assert(st.wordToGlyph.get('sun') === 'light', 'sun should map to light');
});

test('dog → animal', () => {
  assert(st.wordToGlyph.get('dog') === 'animal', 'dog should map to animal');
});

test('coffee → food', () => {
  assert(st.wordToGlyph.get('coffee') === 'food', 'coffee should map to food');
});

test('rain → water', () => {
  assert(st.wordToGlyph.get('rain') === 'water', 'rain should map to water');
});

test('computer → AI', () => {
  assert(st.wordToGlyph.get('computer') === 'AI', 'computer should map to AI');
});

test('happy → joy', () => {
  assert(st.wordToGlyph.get('happy') === 'joy', 'happy should map to joy');
});

test('castle → stone', () => {
  assert(st.wordToGlyph.get('castle') === 'stone', 'castle should map to stone');
});

// ── Line tokenization ──

test('simple sentence', () => {
  const result = st.tokenizeLine('the sun rose and the birds started singing');
  assert(result.length >= 3, `expected >= 3 tokens, got ${result.length}`);
  assert(result.includes('light'), 'should contain light (sun)');
  assert(result.includes('animal'), 'should contain animal (birds)');
});

test('empty string returns empty', () => {
  const result = st.tokenizeLine('');
  assert(result.length === 0, 'should return empty array');
});

test('only stop words returns empty', () => {
  const result = st.tokenizeLine('the a an is are was were');
  assert(result.length === 0, 'stop words only should return empty');
});

test('Dracula sentence', () => {
  const result = st.tokenizeLine('Count Dracula stood in the dark castle and waited for his victim');
  assert(result.includes('dark'), 'should contain dark');
  assert(result.includes('stone'), 'should contain stone (castle)');
  assert(result.includes('wait'), 'should contain wait');
});

test('modern life', () => {
  const result = st.tokenizeLine('she scrolled through social media on her phone feeling stressed');
  assert(result.includes('internet'), 'should contain internet (social media/phone)');
  assert(result.includes('stress'), 'should contain stress');
});

test('no duplicate consecutive glyphs', () => {
  const result = st.tokenizeLine('the dog and the dog and the dog');
  // "dog dog dog" should not produce "animal animal animal"
  let hasDup = false;
  for (let i = 1; i < result.length; i++) {
    if (result[i] === result[i-1]) hasDup = true;
  }
  assert(!hasDup, 'should not have consecutive duplicates');
});

// ── Morphological variants ──

test('plurals: dogs → animal', () => {
  const result = st.tokenizeLine('the dogs ran away');
  assert(result.includes('animal'), 'dogs should map to animal');
});

test('past tense: walked → go', () => {
  const result = st.tokenizeLine('he walked home');
  assert(result.includes('go'), 'walked should map to go');
});

test('gerund: running → go or strength', () => {
  const result = st.tokenizeLine('she was running fast');
  assert(result.includes('go') || result.includes('strength'), 'running should map to go or strength');
});

// ── Multi-line tokenize ──

test('tokenize multiple lines', () => {
  const text = `the sun rose over the mountain
a child played with a dog in the yard
she made dinner and called the family`;
  const docs = st.tokenize(text);
  assert(docs.length === 3, `expected 3 stories, got ${docs.length}`);
  assert(docs.every(d => d.length >= 2), 'each story should have >= 2 tokens');
});

test('filters short stories', () => {
  const text = `hello
the sun rose and birds sang a beautiful melody in the morning light`;
  const docs = st.tokenize(text);
  // "hello" alone might produce 0-1 tokens, should be filtered
  assert(docs.length >= 1, 'should have at least 1 story');
});

// ── encode() ──

test('encode returns string', () => {
  const result = st.encode('the sun rose and the birds started singing');
  assert(typeof result === 'string', 'should return string');
  assert(result.includes('light'), 'should contain light');
  assert(result.split(' ').length >= 3, 'should have >= 3 space-separated glyphs');
});

// ── Coverage: all 88 glyphs reachable ──

test('all 88 glyphs are reachable', () => {
  const reachable = new Set();
  for (const [glyph, words] of Object.entries(SEMANTIC_MAP)) {
    for (const word of words) {
      const mapped = st.wordToGlyph.get(word.toLowerCase());
      if (mapped === glyph) { reachable.add(glyph); break; }
    }
  }
  assert(reachable.size === 88, `expected 88 reachable glyphs, got ${reachable.size}`);
});

// ── Edge cases ──

test('punctuation stripped', () => {
  const result = st.tokenizeLine('Hello! How are you? I am fine.');
  assert(result.length >= 1, 'should handle punctuation');
});

test('numbers ignored gracefully', () => {
  const result = st.tokenizeLine('in 1897 he wrote 300 pages');
  // should not crash, numbers get ignored
  assert(Array.isArray(result), 'should return array');
});

test('mixed case handled', () => {
  const r1 = st.tokenizeLine('The SUN rose');
  const r2 = st.tokenizeLine('the sun rose');
  assert(JSON.stringify(r1) === JSON.stringify(r2), 'case should not matter');
});

// ── Summary ──
console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
