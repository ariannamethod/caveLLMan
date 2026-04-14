import random

EMOJI = ['🌅', '🌙', '☀️', '⭐', '🌧️', '🔥', '🌊', '🌳', '😊', '😢', '😴', '💪', '💕', '🙏', '💡', '👏', '☕', '🍲', '🏃', '📖', '🎵', '📸', '🚗', '🎁', '🏠', '🏥', '🏢', '🗻', '🐕', '👶', '👨‍🍳', '💊']

# Semantic groups
TIME_MORNING = ['🌅', '☀️']
TIME_NIGHT = ['🌙', '⭐']
WEATHER = ['🌧️', '☀️', '🌊']
NATURE = ['🌳', '🌊', '🗻', '🐕']
EMOTIONS_POS = ['😊', '💪', '💕', '🙏', '👏']
EMOTIONS_NEG = ['😢']
ACTIVITIES = ['🏃', '📖', '🎵', '📸', '🚗']
FOOD = ['☕', '🍲', '👨‍🍳']
PLACES = ['🏠', '🏥', '🏢', '🗻']
PEOPLE = ['👶', '🐕']
OBJECTS = ['🎁', '💡', '💊']
REST = ['😴', '🔥']

random.seed(42)

def pick(lst, n=1):
    return [random.choice(lst) for _ in range(n)]

def morning_routine():
    """Morning routine pattern"""
    story = pick(TIME_MORNING)
    story += pick(['☕', '☕', '🍲'], random.randint(1, 2))
    if random.random() > 0.5:
        story += pick(['🏃', '🐕'])
    story += pick(['😊', '💪'])
    if random.random() > 0.5:
        story += pick(['🚗', '🏢'])
    return story

def work_day():
    """Work day pattern"""
    story = pick(TIME_MORNING)
    story += ['☕']
    story += ['🚗'] if random.random() > 0.3 else []
    story += ['🏢']
    story += pick(['💡', '📖', '💪'], random.randint(1, 3))
    if random.random() > 0.5:
        story += pick(['🍲', '☕'])
    story += pick(['💪', '😊', '🙏', '👏'])
    story += pick(TIME_NIGHT) if random.random() > 0.4 else []
    story += ['🏠']
    return story

def adventure():
    """Adventure/travel pattern"""
    story = pick(TIME_MORNING)
    story += ['🚗'] if random.random() > 0.3 else []
    story += pick(['🗻', '🌊', '🌳'], random.randint(1, 3))
    story += pick(['📸', '🏃', '🎵'], random.randint(1, 2))
    story += pick(EMOTIONS_POS, random.randint(1, 2))
    if random.random() > 0.5:
        story += pick(['🔥', '🍲'])
    story += pick(TIME_NIGHT) if random.random() > 0.5 else []
    return story

def love_story():
    """Love/relationship pattern"""
    story = pick(['💕', '😊'])
    story += pick(['🎁', '🍲', '🎵', '☕'], random.randint(2, 4))
    story += ['💕'] * random.randint(1, 2)
    story += pick(['😊', '🙏', '👏'])
    if random.random() > 0.5:
        story += pick(['🌙', '⭐', '🌅'])
    if random.random() > 0.5:
        story += pick(['🏠', '👶'])
    return story

def grief():
    """Grief/sadness pattern"""
    story = pick(['🌧️', '🌙'])
    story += ['😢'] * random.randint(1, 2)
    story += pick(['🙏', '💕', '🏥'], random.randint(1, 2))
    if random.random() > 0.5:
        story += pick(['📖', '🎵', '☕'])
    story += pick(['💪', '🙏', '😊']) if random.random() > 0.3 else ['😢']
    if random.random() > 0.5:
        story += pick(['🌅', '☀️', '⭐'])
    return story

def illness():
    """Illness/recovery pattern"""
    story = pick(['🏥', '💊'])
    story += ['😢'] if random.random() > 0.3 else []
    story += pick(['💊', '🍲', '😴'], random.randint(1, 3))
    story += pick(['🙏', '💕'])
    if random.random() > 0.4:
        story += pick(['💪', '😊', '☀️'])
    if random.random() > 0.5:
        story += ['🏠']
    return story

def celebration():
    """Celebration pattern"""
    story = pick(['🎁', '🎵', '😊'])
    story += pick(['👏', '🎵', '🍲', '🔥'], random.randint(2, 4))
    story += pick(EMOTIONS_POS, random.randint(1, 3))
    if random.random() > 0.4:
        story += pick(['📸', '💕'])
    story += pick(TIME_NIGHT) if random.random() > 0.5 else []
    return story

def nature_walk():
    """Nature/outdoors pattern"""
    story = pick(TIME_MORNING)
    story += pick(['🌳', '🗻', '🌊', '🐕'], random.randint(2, 4))
    story += pick(['🏃', '📸', '🎵'], random.randint(1, 2))
    story += pick(EMOTIONS_POS)
    if random.random() > 0.5:
        story += pick(WEATHER)
    if random.random() > 0.5:
        story += pick(['🔥', '☕', '🍲'])
    return story

def evening_rest():
    """Evening/rest pattern"""
    story = pick(TIME_NIGHT)
    story += pick(['🏠', '🍲', '☕'], random.randint(1, 2))
    story += pick(['📖', '🎵', '🔥'], random.randint(1, 2))
    story += pick(['😴', '😊', '🙏'])
    if random.random() > 0.5:
        story += pick(['💕', '⭐'])
    return story

def rainy_day():
    """Rainy/cozy day pattern"""
    story = ['🌧️']
    story += pick(['🏠', '☕', '📖', '🎵', '🔥'], random.randint(2, 4))
    story += pick(['😊', '😴', '🙏'])
    if random.random() > 0.5:
        story += pick(['🍲', '👨‍🍳'])
    if random.random() > 0.4:
        story += pick(TIME_NIGHT)
    return story

def baby_life():
    """Baby/parenting pattern"""
    story = pick(TIME_MORNING + TIME_NIGHT)
    story += ['👶']
    story += pick(['🍲', '😢', '😴', '💕', '🎵'], random.randint(2, 4))
    story += pick(['💕', '😊', '🙏'])
    if random.random() > 0.5:
        story += pick(['😴', '☕', '💪'])
    return story

def cooking():
    """Cooking pattern"""
    story = pick(['👨‍🍳', '🍲'])
    story += pick(['🔥', '🍲', '👨‍🍳', '💡'], random.randint(2, 4))
    story += pick(EMOTIONS_POS)
    if random.random() > 0.5:
        story += pick(['💕', '🎵', '👏'])
    if random.random() > 0.5:
        story += pick(['🏠', '😊'])
    return story

def fitness():
    """Fitness/exercise pattern"""
    story = pick(TIME_MORNING)
    story += ['🏃']
    story += pick(['💪', '🌊', '🗻', '🏃'], random.randint(1, 3))
    story += pick(['💪', '😊', '👏'])
    story += pick(['☕', '🍲', '😴']) if random.random() > 0.4 else []
    return story

def dog_walk():
    """Dog walking pattern"""
    story = pick(TIME_MORNING + ['☀️'])
    story += ['🐕']
    story += pick(['🌳', '🏃', '🌊', '🗻'], random.randint(1, 3))
    story += pick(['😊', '💕', '📸'])
    if random.random() > 0.5:
        story += pick(['☕', '🏠'])
    return story

def study():
    """Study/learning pattern"""
    story = pick(['☕', '💡'])
    story += ['📖'] * random.randint(1, 2)
    story += pick(['💡', '💪', '🙏'], random.randint(1, 2))
    if random.random() > 0.5:
        story += pick(['☕', '🍲'])
    story += pick(['👏', '😊', '💪'])
    if random.random() > 0.5:
        story += pick(TIME_NIGHT)
    return story

def spiritual():
    """Spiritual/meditation pattern"""
    story = pick(TIME_MORNING + TIME_NIGHT)
    story += pick(['🙏', '⭐', '🌳'], random.randint(1, 2))
    story += pick(['📖', '🎵', '😊'], random.randint(1, 2))
    story += ['🙏']
    if random.random() > 0.5:
        story += pick(['💕', '💡', '☀️'])
    return story

def road_trip():
    """Road trip pattern"""
    story = pick(TIME_MORNING)
    story += ['🚗']
    story += pick(['🎵', '☕', '🗻', '🌊', '🌳', '📸'], random.randint(3, 6))
    story += pick(EMOTIONS_POS, random.randint(1, 2))
    if random.random() > 0.5:
        story += pick(['🔥', '🍲', '⭐'])
    return story

def hospital_visit():
    """Hospital/medical pattern"""
    story = pick(['🏥'])
    story += pick(['💊', '🙏', '😢', '💪'], random.randint(2, 3))
    if random.random() > 0.4:
        story += pick(['💕', '👏', '😊'])
    story += pick(['🏠', '☀️', '🙏'])
    return story

def music_night():
    """Music/creative night"""
    story = pick(TIME_NIGHT)
    story += ['🎵'] * random.randint(1, 2)
    story += pick(['🔥', '💡', '😊', '💕'], random.randint(1, 3))
    story += pick(['👏', '⭐', '😊'])
    if random.random() > 0.5:
        story += pick(['🍲', '☕', '🏠'])
    return story

def full_day():
    """Full day pattern (long)"""
    story = pick(TIME_MORNING)
    story += pick(['☕', '🍲'])
    story += pick(['🚗', '🏢', '🏃'])
    story += pick(['💡', '📖', '💪'], random.randint(1, 2))
    story += pick(['🍲', '☕'])
    story += pick(['💪', '👏', '😊'])
    story += pick(['🚗', '🏠'])
    story += pick(['🍲', '🎵', '📖'])
    story += pick(['😴', '🌙'])
    return story

def weekend():
    """Weekend pattern (long)"""
    story = pick(TIME_MORNING)
    story += ['😴'] if random.random() > 0.5 else []
    story += pick(['☕', '🍲'])
    story += pick(['🏃', '🐕', '🌳', '📸', '🎵'], random.randint(2, 4))
    story += pick(EMOTIONS_POS)
    story += pick(['🍲', '👨‍🍳', '🔥'])
    story += pick(['💕', '😊', '🎵'])
    story += pick(TIME_NIGHT)
    story += ['😴']
    return story

def storm():
    """Storm/difficult times"""
    story = ['🌧️'] * random.randint(1, 2)
    story += pick(['🌊', '😢', '🏠'], random.randint(1, 2))
    story += pick(['🔥', '☕', '📖', '🎵'])
    story += pick(['💪', '🙏'])
    if random.random() > 0.4:
        story += pick(['☀️', '🌅', '😊'])
    return story

def gift_giving():
    """Gift/generosity pattern"""
    story = pick(['😊', '💡'])
    story += ['🎁']
    story += pick(['💕', '😊', '👏', '🙏'], random.randint(2, 3))
    if random.random() > 0.5:
        story += pick(['🍲', '🎵', '☕'])
    story += pick(['💕', '😊'])
    return story

# Extended templates for more variety
def random_coherent_short():
    """Generate a short coherent story from transitions"""
    transitions = {
        '🌅': ['☕', '😊', '🏃', '🐕', '☀️', '💪'],
        '🌙': ['⭐', '😴', '📖', '🎵', '🏠', '🙏'],
        '☀️': ['😊', '🏃', '🌊', '🌳', '🐕', '📸'],
        '⭐': ['🌙', '😴', '🙏', '💕', '😊', '📖'],
        '🌧️': ['🏠', '☕', '📖', '😢', '🔥', '🎵'],
        '🔥': ['🍲', '😊', '🌙', '⭐', '☕', '🎵'],
        '🌊': ['🏃', '😊', '📸', '🗻', '🌳', '💪'],
        '🌳': ['🐕', '🏃', '📸', '😊', '🌊', '🗻'],
        '😊': ['💕', '👏', '🎵', '🙏', '💪', '📸', '☀️'],
        '😢': ['🙏', '💕', '💊', '😴', '🏥', '☕'],
        '😴': ['🌅', '🌙', '💪', '☀️', '😊', '🏠'],
        '💪': ['🏃', '😊', '👏', '💡', '🗻', '🌊'],
        '💕': ['😊', '🎁', '🙏', '💕', '👶', '🏠'],
        '🙏': ['😊', '💪', '💕', '⭐', '☀️', '🌅'],
        '💡': ['📖', '💪', '👏', '😊', '🏢', '🔥'],
        '👏': ['😊', '💪', '🎵', '💕', '🙏', '🎁'],
        '☕': ['📖', '💡', '🏢', '😊', '🚗', '🍲'],
        '🍲': ['😊', '💕', '👨‍🍳', '🔥', '🏠', '💪'],
        '🏃': ['💪', '🌊', '🗻', '🐕', '😊', '☀️'],
        '📖': ['💡', '☕', '😊', '🌙', '📖', '🙏'],
        '🎵': ['😊', '💕', '🌙', '🔥', '👏', '💡'],
        '📸': ['😊', '🌊', '🗻', '🌳', '📸', '💕'],
        '🚗': ['🏢', '🗻', '🌊', '🎵', '🏠', '☕'],
        '🎁': ['😊', '💕', '👏', '🙏', '💕', '🎵'],
        '🏠': ['😴', '☕', '🍲', '📖', '🎵', '💕'],
        '🏥': ['💊', '🙏', '😢', '💪', '💕', '🏠'],
        '🏢': ['💡', '☕', '📖', '💪', '🚗', '🏠'],
        '🗻': ['📸', '😊', '💪', '🌊', '🌳', '⭐'],
        '🐕': ['🏃', '🌳', '😊', '💕', '🏠', '📸'],
        '👶': ['💕', '😢', '😴', '🍲', '🎵', '😊'],
        '👨‍🍳': ['🍲', '🔥', '😊', '👏', '💡', '🏠'],
        '💊': ['🙏', '😴', '💪', '🏥', '🍲', '☀️'],
    }
    start = random.choice(EMOJI)
    length = random.randint(6, 8)
    story = [start]
    for _ in range(length - 1):
        current = story[-1]
        nexts = transitions.get(current, EMOJI)
        story.append(random.choice(nexts))
    return story

def random_coherent_medium():
    """Medium coherent story using transition chains"""
    transitions = {
        '🌅': ['☕', '😊', '🏃', '🐕', '☀️', '💪', '🌳'],
        '🌙': ['⭐', '😴', '📖', '🎵', '🏠', '🙏', '💕'],
        '☀️': ['😊', '🏃', '🌊', '🌳', '🐕', '📸', '💪'],
        '⭐': ['🌙', '😴', '🙏', '💕', '😊', '📖', '🎵'],
        '🌧️': ['🏠', '☕', '📖', '😢', '🔥', '🎵', '🌊'],
        '🔥': ['🍲', '😊', '🌙', '⭐', '☕', '🎵', '👨‍🍳'],
        '🌊': ['🏃', '😊', '📸', '🗻', '🌳', '💪', '🌅'],
        '🌳': ['🐕', '🏃', '📸', '😊', '🌊', '🗻', '⭐'],
        '😊': ['💕', '👏', '🎵', '🙏', '💪', '📸', '☀️', '🏠'],
        '😢': ['🙏', '💕', '💊', '😴', '🏥', '☕', '🌧️'],
        '😴': ['🌅', '🌙', '💪', '☀️', '😊', '🏠', '⭐'],
        '💪': ['🏃', '😊', '👏', '💡', '🗻', '🌊', '☀️'],
        '💕': ['😊', '🎁', '🙏', '💕', '👶', '🏠', '🎵'],
        '🙏': ['😊', '💪', '💕', '⭐', '☀️', '🌅', '💡'],
        '💡': ['📖', '💪', '👏', '😊', '🏢', '🔥', '🎵'],
        '👏': ['😊', '💪', '🎵', '💕', '🙏', '🎁', '🔥'],
        '☕': ['📖', '💡', '🏢', '😊', '🚗', '🍲', '☀️'],
        '🍲': ['😊', '💕', '👨‍🍳', '🔥', '🏠', '💪', '😴'],
        '🏃': ['💪', '🌊', '🗻', '🐕', '😊', '☀️', '🏠'],
        '📖': ['💡', '☕', '😊', '🌙', '📖', '🙏', '💪'],
        '🎵': ['😊', '💕', '🌙', '🔥', '👏', '💡', '😴'],
        '📸': ['😊', '🌊', '🗻', '🌳', '📸', '💕', '👏'],
        '🚗': ['🏢', '🗻', '🌊', '🎵', '🏠', '☕', '🌳'],
        '🎁': ['😊', '💕', '👏', '🙏', '💕', '🎵', '👶'],
        '🏠': ['😴', '☕', '🍲', '📖', '🎵', '💕', '🔥'],
        '🏥': ['💊', '🙏', '😢', '💪', '💕', '🏠', '☀️'],
        '🏢': ['💡', '☕', '📖', '💪', '🚗', '🏠', '🍲'],
        '🗻': ['📸', '😊', '💪', '🌊', '🌳', '⭐', '🏃'],
        '🐕': ['🏃', '🌳', '😊', '💕', '🏠', '📸', '🌊'],
        '👶': ['💕', '😢', '😴', '🍲', '🎵', '😊', '🙏'],
        '👨‍🍳': ['🍲', '🔥', '😊', '👏', '💡', '🏠', '💪'],
        '💊': ['🙏', '😴', '💪', '🏥', '🍲', '☀️', '😊'],
    }
    start = random.choice(EMOJI)
    length = random.randint(9, 14)
    story = [start]
    for _ in range(length - 1):
        current = story[-1]
        nexts = transitions.get(current, EMOJI)
        story.append(random.choice(nexts))
    return story

# Template generators with their target length categories
TEMPLATES_SHORT = [morning_routine, evening_rest, grief, illness, hospital_visit, storm, gift_giving, dog_walk, spiritual]
TEMPLATES_MEDIUM = [work_day, adventure, love_story, celebration, nature_walk, rainy_day, baby_life, cooking, fitness, study, music_night, road_trip]
TEMPLATES_LONG = [full_day, weekend]

def generate_story(target_length):
    """Generate a story of target length category: 'short', 'medium', 'long'"""
    if target_length == 'short':
        if random.random() < 0.4:
            story = random_coherent_short()
        else:
            template = random.choice(TEMPLATES_SHORT)
            story = template()
            # Trim or pad to 6-8
            while len(story) < 6:
                story.append(random.choice(EMOTIONS_POS + ['😴', '🏠']))
            if len(story) > 8:
                story = story[:8]
    elif target_length == 'medium':
        if random.random() < 0.35:
            story = random_coherent_medium()
        else:
            template = random.choice(TEMPLATES_MEDIUM)
            story = template()
            # Trim or pad to 9-14
            while len(story) < 9:
                story.append(random.choice(EMOTIONS_POS + REST + FOOD))
            if len(story) > 14:
                story = story[:14]
    else:  # long
        if random.random() < 0.3:
            # Chain two short templates
            t1 = random.choice(TEMPLATES_SHORT + TEMPLATES_MEDIUM)
            t2 = random.choice(TEMPLATES_SHORT + TEMPLATES_MEDIUM)
            story = t1() + t2()
        else:
            template = random.choice(TEMPLATES_LONG + TEMPLATES_MEDIUM)
            story = template()
        # Pad to 15-20
        while len(story) < 15:
            story.append(random.choice(EMOJI))
        if len(story) > 20:
            story = story[:20]

    return story

def main():
    stories = set()
    # Distribution: 30% short, 40% medium, 30% long
    targets = ['short'] * 3000 + ['medium'] * 4000 + ['long'] * 3000
    random.shuffle(targets)

    results = []
    attempts = 0
    max_attempts = 50000

    for i, target in enumerate(targets):
        while attempts < max_attempts:
            attempts += 1
            story = generate_story(target)
            line = ' '.join(story)
            if line not in stories:
                stories.add(line)
                results.append(line)
                break
        else:
            # Fallback: force unique by adding random suffix
            story = generate_story(target)
            story.append(random.choice(EMOJI))
            line = ' '.join(story)
            results.append(line)

    # Verify we have 10000
    assert len(results) == 10000, f"Got {len(results)} stories"

    with open('/tmp/emolm_stories_10k.txt', 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')

    # Stats
    lengths = [len(line.split(' ')) for line in results]
    short = sum(1 for l in lengths if 6 <= l <= 8)
    medium = sum(1 for l in lengths if 9 <= l <= 14)
    long = sum(1 for l in lengths if 15 <= l <= 20)
    print(f"Generated {len(results)} unique stories")
    print(f"Short (6-8): {short} ({short/100:.1f}%)")
    print(f"Medium (9-14): {medium} ({medium/100:.1f}%)")
    print(f"Long (15-20): {long} ({long/100:.1f}%)")
    print(f"Min length: {min(lengths)}, Max length: {max(lengths)}")
    print(f"Unique: {len(set(results))}")

if __name__ == '__main__':
    main()
