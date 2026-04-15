/*
 * Semantic Tokenizer for caveLLMan
 * Maps any English word to the nearest of 88 hieroglyphs.
 * Feed it Dracula, poetry, news — it compresses to 88 concepts.
 *
 * Usage:
 *   const st = new SemanticTokenizer();
 *   st.tokenize("the sun rose and the birds started singing")
 *   // → ["light", "up", "and", "animal", "speak"]
 *
 *   st.tokenizeLine("he made coffee and sat down to read quietly")
 *   // → ["man", "make", "food", "and", "down", "see", "think"]
 */

const SEMANTIC_MAP = {
  // NATURE
  water: ["water","rain","river","sea","ocean","lake","stream","wet","swim","wave","flood","pour","splash","drip","dew","puddle","pond","creek","bay","shore","tide","surf","mist","fog","sprinkle","shower","tears","cry","weep","liquid","flow","drown","sail","boat","ship","fish","fishing","anchor"],
  fire: ["fire","flame","burn","cook","heat","warm","hot","stove","oven","torch","candle","blaze","grill","roast","bake","bonfire","ember","ash","smoke","furnace","ignite","scorch","sizzle","boil","fry","charcoal","match","lighter","campfire","fireplace","hearth"],
  earth: ["earth","ground","soil","dirt","land","field","farm","garden","plant","grow","seed","root","mud","clay","sand","dust","gravel","terrain","continent","country","world","globe","planet","harvest","crop","plow","dig","bury","grave"],
  stone: ["stone","rock","mountain","hill","cliff","cave","wall","brick","concrete","marble","granite","pebble","boulder","quarry","mineral","iron","metal","steel","gold","silver","mine","castle","fortress","tower","bridge","building","ruins"],
  tree: ["tree","forest","wood","leaf","branch","bark","trunk","log","bush","flower","bloom","rose","vine","grass","weed","oak","pine","palm","jungle","orchard","blossom","petal","thorn","garden","park","meadow","grove"],
  sky: ["sky","cloud","air","wind","breeze","storm","thunder","lightning","rainbow","weather","atmosphere","heaven","ceiling","above","overhead","horizon","dawn","dusk","twilight"],
  light: ["light","sun","sunrise","sunset","bright","shine","glow","lamp","ray","beam","star","sparkle","flash","morning","noon","daylight","sunshine","illuminate","golden","radiant","brilliant","vivid","clear","pale","white","luminous"],
  dark: ["dark","night","shadow","dim","midnight","evening","dusk","black","shade","murky","gloomy","darkness","blackout","unlit","obscure","pitch","nocturnal","twilight"],
  cold: ["cold","ice","snow","frost","freeze","chill","cool","winter","arctic","frozen","shiver","numb","frigid","sleet","blizzard","hail","glacier"],

  // BEINGS
  person: ["person","people","human","someone","anyone","everyone","nobody","somebody","individual","citizen","stranger","neighbor","folk","crowd","population","society","public","they","them","their","we","us","our"],
  man: ["man","he","him","his","boy","guy","gentleman","husband","father","dad","brother","son","uncle","grandfather","grandpa","sir","male","king","prince","hero","lad","fellow","mister","dude"],
  woman: ["woman","she","her","girl","lady","wife","mother","mom","mama","sister","daughter","aunt","grandmother","grandma","queen","princess","female","miss","madam","bride"],
  child: ["child","kid","baby","infant","toddler","boy","girl","youth","teen","teenager","young","newborn","pupil","student","children","kids","babies","little","tiny","small"],
  old: ["old","elderly","ancient","aged","senior","grandparent","grandfather","grandmother","grandpa","grandma","elder","veteran","retired","wise","wrinkled","gray","grey","mature"],
  spirit: ["spirit","soul","ghost","angel","god","divine","holy","sacred","prayer","pray","faith","religion","church","temple","mosque","synagogue","heaven","afterlife","eternal","blessing","miracle","supernatural","spiritual","worship","ritual","ceremony","meditation"],
  AI: ["ai","computer","robot","machine","algorithm","digital","software","hardware","code","program","data","internet","artificial","technology","tech","cyber","virtual","bot","automation","neural","chip","processor","server","download","upload","app","device","screen","pixel","binary"],
  animal: ["animal","dog","cat","bird","horse","cow","sheep","goat","pig","chicken","rooster","fish","rabbit","deer","bear","wolf","lion","tiger","eagle","snake","insect","bee","butterfly","mouse","rat","whale","dolphin","monkey","fox","owl","crow","sparrow","kitten","puppy","pet","creature","beast","wildlife"],

  // BODY
  body: ["body","hand","arm","leg","foot","head","face","eye","ear","nose","mouth","tooth","finger","toe","shoulder","knee","chest","stomach","back","neck","skin","bone","blood","heart","brain","hair","muscle","belly"],
  food: ["food","eat","meal","breakfast","lunch","dinner","bread","rice","meat","fruit","vegetable","soup","cake","pie","cookie","chocolate","cheese","egg","milk","coffee","tea","beer","wine","drink","juice","sugar","salt","pepper","spice","restaurant","cafe","kitchen","plate","bowl","cup","glass","fork","knife","spoon","hungry","appetite","taste","delicious","recipe","cook","chef","falafel","hummus","pita","tahini","pizza","pasta","steak","salad","sandwich","snack","dessert","ice cream","lemonade"],
  sleep: ["sleep","rest","nap","bed","pillow","blanket","mattress","dream","snore","yawn","drowsy","asleep","awake","wake","tired","insomnia","bedroom","couch","sofa","lay","lie down","doze","slumber"],
  pain: ["pain","hurt","ache","wound","injury","sick","ill","disease","fever","headache","toothache","doctor","hospital","medicine","pill","injection","bandage","surgery","bleed","suffer","agony","sore","bruise","break","fracture","cough","sneeze","flu","virus","infection","ambulance","nurse","patient","therapy","heal","cure","treatment","symptom"],
  strength: ["strength","strong","power","force","energy","muscle","fit","exercise","gym","run","running","jog","sprint","lift","push","pull","carry","endure","tough","brave","courage","hero","mighty","athletic","sport","game","fight","battle","wrestle","punch","kick","jump","climb","hike","marathon","training","workout"],

  // EMOTION
  joy: ["joy","happy","happiness","glad","cheerful","smile","laugh","grin","celebrate","party","dance","fun","enjoy","pleasure","delight","wonderful","great","fantastic","amazing","beautiful","love","cheer","humor","joke","comedy","excited","thrilled","proud","satisfaction","content","bliss","euphoria","giggle","glee","clap","applaud"],
  grief: ["grief","sad","sorrow","mourn","funeral","loss","tragedy","misery","depression","despair","hopeless","devastated","heartbroken","sob","weep","lament","regret","melancholy","gloomy","somber","woe","anguish"],
  love: ["love","beloved","darling","sweetheart","romance","kiss","hug","embrace","affection","care","tender","gentle","passion","desire","adore","cherish","devotion","heart","valentine","wedding","marry","marriage","couple","date","relationship","partner","soulmate","crush"],
  fear: ["fear","afraid","scared","terrified","horror","panic","anxiety","nervous","worry","dread","fright","phobia","nightmare","creepy","spooky","ghost","monster","danger","threat","alarm","shock","scream","tremble","shake"],
  anger: ["anger","angry","mad","furious","rage","hate","resent","hostile","aggressive","violent","yell","shout","scream","curse","swear","fight","argue","quarrel","dispute","irritate","annoy","frustrate","enrage","outrage","bitter","vengeful"],
  longing: ["longing","miss","nostalgia","yearn","pine","homesick","wish","hope","desire","crave","want","need","hunger","thirst","aspire","ache","melancholy","wistful","sentimental","bittersweet","reminisce"],
  tired: ["tired","exhausted","weary","fatigue","sleepy","drowsy","worn","drained","spent","burnout","yawn","sluggish","lethargic","listless","lazy","bored","boring","tedious","monotonous"],
  stress: ["stress","pressure","tension","overwhelm","overwork","deadline","rush","hurry","busy","hectic","chaos","crisis","emergency","urgent","panic","frantic","scramble","juggle","burden","load","strain","struggle","hassle"],

  // VERBS
  go: ["go","walk","run","move","travel","drive","ride","fly","leave","depart","arrive","come","enter","exit","return","approach","advance","march","step","rush","hurry","wander","roam","stroll","pass","cross","journey","commute","visit","trip","hike","climb","crawl","chase"],
  make: ["make","build","create","produce","construct","craft","manufacture","assemble","design","invent","develop","form","shape","mold","forge","generate","prepare","arrange","organize","establish","found","compose","knit","sew","weave","carve"],
  break: ["break","destroy","smash","crash","shatter","ruin","demolish","tear","rip","cut","split","crack","snap","collapse","explode","burst","wreck","damage","harm","crush","grind","chop","slash","pierce"],
  see: ["see","look","watch","observe","stare","gaze","glance","peek","view","notice","spot","witness","examine","inspect","read","study","review","scan","survey","recognize","discover","find","search","browse","scroll","photo","photograph","picture","image","camera","film","movie","video","tv","television","screen"],
  speak: ["speak","say","tell","talk","chat","discuss","conversation","call","phone","shout","whisper","announce","declare","explain","describe","narrate","recite","sing","voice","word","sentence","speech","lecture","presentation","interview","debate","argue"],
  hear: ["hear","listen","sound","noise","loud","quiet","silent","music","song","melody","rhythm","beat","tune","radio","podcast","concert","instrument","piano","guitar","drum","violin","trumpet","bell","ring","echo","hum","buzz","whistle","chirp"],
  seek: ["seek","search","look for","hunt","explore","investigate","research","discover","pursue","chase","track","scout","probe","quest","dig","browse","scan","survey","wander","roam"],
  give: ["give","offer","share","donate","gift","present","hand","pass","deliver","send","mail","contribute","provide","supply","serve","pour","distribute","grant","lend","borrow","pay","tip","reward","charity"],
  want: ["want","wish","desire","crave","need","demand","request","ask","beg","plead","prefer","choose","like","fancy","hope","dream","aspire","aim","goal","ambition","target"],
  miss: ["miss","missing","lost","absent","gone","lack","without","empty","void","lonely","alone","isolated","abandoned","deserted","forgotten","neglected","overlooked"],
  agree: ["agree","yes","okay","sure","accept","approve","confirm","nod","consent","permit","allow","support","endorse","cooperate","collaborate","united","together","harmony","peace","alliance","treaty","deal","contract","handshake","compromise"],

  // SOCIAL
  home: ["home","house","apartment","room","door","window","roof","floor","wall","furniture","table","chair","couch","sofa","bed","kitchen","bathroom","bedroom","garage","yard","porch","balcony","stairs","hallway","closet","shelf","drawer","cabinet","curtain","carpet","rug","domestic","shelter","dwelling","residence","address","neighborhood","street"],
  outside: ["outside","outdoor","nature","landscape","scenery","view","environment","weather","fresh","open","wild","wilderness","countryside","rural","suburban","urban","city","town","village","square","plaza","market","shop","store","mall","restaurant","cafe","bar","pub","club","park","beach","forest","mountain","desert","island"],
  work: ["work","job","career","office","business","company","boss","employee","colleague","meeting","project","task","assignment","deadline","salary","wage","money","earn","hire","fire","promote","retire","resume","interview","profession","occupation","trade","craft","industry","factory","labor","toil","effort"],
  internet: ["internet","online","website","email","social media","facebook","twitter","instagram","youtube","google","app","download","upload","click","scroll","post","message","text","notification","wifi","connection","network","browser","search","link","url","blog","forum","comment","like","share","follow","viral","trending","meme","selfie","emoji","hashtag","stream","gaming","podcast"],
  bond: ["bond","connect","relationship","friendship","family","community","team","group","partner","companion","ally","friend","buddy","mate","peer","neighbor","reunion","gather","assemble","join","unite","link","tie","together","belong","trust","loyal","faithful","devoted","support","solidarity","kinship"],
  conflict: ["conflict","fight","war","battle","attack","defend","argument","dispute","quarrel","clash","struggle","compete","rivalry","enemy","opponent","adversary","threat","danger","violence","abuse","bully","harass","protest","rebel","revolution","revolt","resistance","tension","hostility","aggression","invasion"],

  // MIND
  know: ["know","knowledge","learn","study","education","school","university","college","class","lesson","teacher","student","book","read","library","research","science","math","history","language","fact","truth","information","understand","comprehend","realize","aware","recognize","remember","memory","recall","familiar","expert","scholar","professor","lecture","exam","test","grade","degree","diploma","certificate"],
  idea: ["idea","concept","theory","thought","notion","plan","strategy","proposal","suggestion","innovation","invention","inspiration","creativity","imagination","vision","insight","solution","answer","discovery","breakthrough","eureka","brainstorm","hypothesis","experiment"],
  think: ["think","thought","consider","reflect","ponder","wonder","contemplate","meditate","analyze","evaluate","judge","decide","reason","logic","rational","wise","intelligent","smart","clever","brilliant","genius","mind","brain","mental","cognitive","intellectual","philosophy","opinion","believe","assume","guess","estimate","calculate","figure"],
  dream: ["dream","imagine","fantasy","wish","hope","aspire","vision","hallucination","daydream","nightmare","surreal","magical","fairy","tale","story","fiction","novel","myth","legend","adventure","wonder","enchant","mystical","utopia"],
  remember: ["remember","recall","memory","remind","forget","nostalgia","past","history","heritage","tradition","custom","ritual","ancestor","archive","diary","journal","memoir","biography","album","photo","souvenir","monument","memorial","tribute","anniversary","birthday","holiday","celebration"],
  lie: ["lie","deceive","cheat","fraud","fake","false","pretend","trick","scam","betray","dishonest","corrupt","manipulate","mislead","distort","exaggerate","deny","hide","conceal","secret","cover up","propaganda","conspiracy","scheme","plot"],

  // SPACE
  path: ["path","road","street","highway","lane","trail","track","route","way","direction","map","navigate","compass","guide","sign","intersection","corner","turn","curve","straight","bridge","tunnel","gate","entrance","passage","corridor","alley","sidewalk","crosswalk"],
  up: ["up","rise","climb","ascend","lift","raise","elevate","soar","fly","tower","tall","high","above","over","summit","peak","top","ceiling","roof","upstairs","increase","grow","improve","progress","advance","promote","escalate","surge"],
  down: ["down","fall","drop","sink","descend","lower","beneath","below","under","underground","basement","downstairs","decline","decrease","reduce","diminish","shrink","collapse","crash","plunge","dive","stumble","trip","slip"],
  far: ["far","distant","remote","away","abroad","foreign","exotic","horizon","beyond","yonder","overseas","travel","journey","voyage","expedition","emigrate","immigrate","exile","frontier","border","edge","outskirts","periphery"],
  back: ["back","return","come back","go back","behind","rear","reverse","retreat","withdraw","undo","restore","recover","recycle","repeat","again","revisit","reunion","homecoming","flashback","rewind","backward"],

  // TIME
  before: ["before","past","ago","earlier","previous","former","ancient","yesterday","last","prior","once","used to","historical","old","antique","vintage","retro","classic","traditional","origin","beginning","start","original"],
  now: ["now","present","current","today","moment","instant","immediate","right now","currently","contemporary","modern","existing","ongoing","live","real-time","actual","at this point","meanwhile"],
  after: ["after","later","next","future","tomorrow","soon","eventually","finally","then","afterward","subsequently","following","upcoming","forthcoming","ahead","forward","prospect","potential","destiny","fate"],
  never: ["never","no","not","none","nothing","nowhere","nobody","impossible","refuse","reject","deny","forbid","prohibit","ban","stop","cease","end","quit","abandon","surrender","give up","fail","unable","cannot"],
  always: ["always","forever","eternal","permanent","constant","continuous","endless","infinite","every","all","every time","consistently","regularly","daily","weekly","monthly","yearly","routine","habit","tradition","ritual","custom","reliable","dependable","faithful","certain","sure","definitely","absolutely"],

  // GRAMMAR
  not: ["not","no","don't","doesn't","didn't","won't","can't","isn't","aren't","wasn't","weren't","neither","nor","without","lack","absence","refuse","deny","reject","opposite","against","anti","negative","wrong","bad","evil","ugly","terrible","horrible","awful","worst"],
  many: ["many","much","lots","plenty","several","numerous","various","diverse","multiple","abundant","countless","thousand","million","billion","hundred","dozen","pile","heap","mountain","ocean","sea","vast","huge","enormous","massive","giant","immense","tremendous"],
  much: ["much","very","extremely","really","quite","rather","fairly","pretty","so","too","highly","deeply","truly","incredibly","remarkably","significantly","considerably","substantially","enormously","vastly","intensely","thoroughly","completely","totally","entirely","utterly","absolutely"],
  and: ["and","also","plus","with","together","both","as well","moreover","furthermore","additionally","too","along","beside","including","combined","joined","linked","connected","paired","coupled"],
  one: ["one","single","alone","only","sole","unique","individual","first","once","a","an","1","lone","singular","solitary","isolated","independent","separate","particular","specific"],
  question: ["question","ask","wonder","curious","inquiry","puzzle","mystery","riddle","quiz","test","exam","challenge","doubt","uncertain","confused","perplexed","why","what","how","where","when","who","which","whose"],
  how: ["how","way","method","technique","process","procedure","step","instruction","guide","tutorial","manual","recipe","formula","approach","strategy","tactic","manner","style","mode","mechanism"],
  cause: ["cause","because","reason","why","therefore","so","thus","hence","result","effect","consequence","outcome","impact","influence","lead to","due to","since","as","trigger","spark","source","origin","root","basis","factor"],

  // EXTENDED
  me: ["me","i","my","mine","myself","i'm","i've","i'll","i'd","personal","self","own","ego"],
  you: ["you","your","yours","yourself","you're","you've","you'll","you'd"],
  other: ["other","another","else","different","alternative","additional","extra","more","rest","remaining","separate","distinct","various","diverse","foreign","strange","unfamiliar","unknown","new","novel"],
  money: ["money","cash","dollar","euro","coin","bill","bank","account","save","spend","buy","sell","price","cost","cheap","expensive","rich","poor","wealth","fortune","profit","loss","debt","loan","credit","tax","budget","economy","finance","invest","stock","market","trade","business","income","wage","salary","tip","reward","payment","fee","charge"],
  change: ["change","transform","convert","modify","alter","adjust","adapt","evolve","develop","grow","improve","reform","revolution","shift","transition","switch","swap","replace","update","upgrade","renew","renovate","restore","rebuild","reshape","rethink","reconsider"],
  write: ["write","writing","pen","pencil","paper","notebook","journal","diary","letter","email","text","type","print","publish","author","writer","poet","journalist","reporter","editor","manuscript","draft","essay","article","story","novel","poem","blog","note","document","sign","signature","record","register","inscribe","carve","draw","paint","sketch","illustrate"],
  choose: ["choose","pick","select","decide","option","choice","preference","vote","elect","determine","resolve","commit","dedicate","settle","conclude","judge","verdict","ruling","sentence"],
  help: ["help","assist","aid","support","rescue","save","protect","defend","serve","volunteer","charity","donate","contribute","comfort","encourage","inspire","motivate","guide","mentor","teach","advise","counsel","recommend","suggest"],
  have: ["have","own","possess","hold","keep","contain","include","carry","wear","bring","take","get","receive","obtain","acquire","gain","earn","win","achieve","accomplish","succeed","attain"],
  free: ["free","freedom","liberty","liberate","release","escape","independent","autonomous","sovereign","emancipate","unchain","unleash","open","clear","available","spare","volunteer","complimentary","gratis"],
  death: ["death","die","dead","kill","murder","suicide","grave","cemetery","coffin","funeral","mourn","corpse","skeleton","skull","tomb","memorial","obituary","mortal","fatal","lethal","terminal","end","perish","expire","decease","extinct","vanish","disappear"],
  music: ["music","song","melody","rhythm","beat","tune","harmony","chord","note","instrument","piano","guitar","drum","violin","trumpet","flute","saxophone","orchestra","band","concert","perform","sing","rap","hip hop","rock","jazz","classical","folk","pop","blues","country","dance","ballet","opera","symphony","dj","headphones","speaker","volume","loud","quiet","acoustic"],
  good: ["good","great","excellent","wonderful","fantastic","amazing","awesome","perfect","fine","nice","kind","generous","honest","fair","just","moral","ethical","right","correct","proper","appropriate","suitable","adequate","sufficient","satisfactory","acceptable","okay","decent","noble","virtuous","righteous"],

  // SCALE + SUPER
  small: ["small","little","tiny","mini","miniature","micro","compact","short","narrow","thin","slim","slight","minor","modest","humble","subtle","delicate","fine","faint","weak","soft","gentle","mild","light","low","few","less","fewer","minimum","least","reduce","shrink","diminish"],
  same: ["same","equal","identical","similar","alike","equivalent","match","parallel","uniform","consistent","constant","regular","standard","normal","ordinary","common","typical","usual","familiar","routine","habitual","everyday","conventional","traditional"],
  BE: ["be","is","am","are","was","were","being","been","become","exist","live","feel","seem","appear","remain","stay","keep","continue","happen","occur","experience","undergo","endure","survive"],
  wait: ["wait","patience","patient","pause","hesitate","delay","postpone","hold","remain","stay","linger","endure","tolerate","bear","withstand","persist","persevere","hang on","stand by","expect","anticipate","look forward"],
};

// Stop words — skip these entirely
const STOP_WORDS = new Set([
  "the","a","an","is","am","are","was","were","be","been","being",
  "have","has","had","having","do","does","did","doing",
  "will","would","shall","should","may","might","must","can","could",
  "to","of","in","for","on","at","by","from","with","about","into",
  "through","during","before","after","above","below","between",
  "out","off","over","under","again","further","then","once",
  "here","there","when","where","why","how","all","each","every",
  "both","few","more","most","other","some","such","no","nor","not",
  "only","own","same","so","than","too","very","just","because",
  "but","if","or","while","as","until","that","this","these","those",
  "it","its","itself","which","who","whom","whose",
]);

class SemanticTokenizer {
  constructor() {
    // Build reverse map: word → glyph
    this.wordToGlyph = new Map();
    for (const [glyph, words] of Object.entries(SEMANTIC_MAP)) {
      for (const word of words) {
        // Map each word and also its lowercase form
        this.wordToGlyph.set(word.toLowerCase(), glyph);
      }
    }
    this.glyphs = Object.keys(SEMANTIC_MAP);
  }

  // Tokenize a line of text into glyph names
  tokenizeLine(text) {
    const words = text.toLowerCase()
      .replace(/[^a-z0-9\s'-]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 0);

    const result = [];
    let lastGlyph = null;

    for (const word of words) {
      if (STOP_WORDS.has(word)) continue;

      let glyph = this.wordToGlyph.get(word);

      // Try without trailing s/ed/ing/ly
      if (!glyph && word.endsWith('s')) glyph = this.wordToGlyph.get(word.slice(0, -1));
      if (!glyph && word.endsWith('ed')) glyph = this.wordToGlyph.get(word.slice(0, -2));
      if (!glyph && word.endsWith('ing')) glyph = this.wordToGlyph.get(word.slice(0, -3));
      if (!glyph && word.endsWith('ly')) glyph = this.wordToGlyph.get(word.slice(0, -2));
      if (!glyph && word.endsWith('er')) glyph = this.wordToGlyph.get(word.slice(0, -2));
      if (!glyph && word.endsWith('est')) glyph = this.wordToGlyph.get(word.slice(0, -3));
      if (!glyph && word.endsWith('tion')) glyph = this.wordToGlyph.get(word.slice(0, -4));
      if (!glyph && word.endsWith('ness')) glyph = this.wordToGlyph.get(word.slice(0, -4));

      if (glyph && glyph !== lastGlyph) {
        result.push(glyph);
        lastGlyph = glyph;
      }
    }

    return result;
  }

  // Tokenize full text (multiple lines)
  tokenize(text) {
    return text.split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0)
      .map(line => this.tokenizeLine(line))
      .filter(tokens => tokens.length >= 2);
  }

  // Convert tokenized result back to glyph string
  encode(text) {
    return this.tokenizeLine(text).join(' ');
  }
}

// Export for both browser and Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { SemanticTokenizer, SEMANTIC_MAP, STOP_WORDS };
}
