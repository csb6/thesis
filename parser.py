import time, os, re, itertools, spacy, traceback
from tzwhere import tzwhere

os.environ["TZ"] = "US/Pacific"
time.tzset()

tz = tzwhere.tzwhere()

en_lang = spacy.load("en_core_web_sm")

# Row 0
Username_Col = 0
Name_Col = 1
User_ID_Col = 2
User_Location_Col = 3
User_UTC_Offset_Col = 5
User_Timezone_Col = 6
User_Creation_Time_Col = 7
User_Lang_Col = 8

# Row 1
Post_Time_Col = 0
Post_Timezone_Col = 1 # Calculated from post coordinates
Post_City_Coords_Col = 2

# Convert a string field from the text file to its value or to None
def get(field):
    if field == "NIL":
        return None
    else:
        return field

# Convert all fields in row to their values (or to None)
def get_all(row):
    return [get(cell) for cell in row.strip().split("\t")]

# Converts string of coordinates to a timezone
def to_timezone(lat_long_str):
    if not lat_long_str:
        return None
    try:
        latitude, longitude = [float(n) for n in lat_long_str.split("|", 2)]
        return tz.tzNameAt(latitude, longitude)
    except:
        print("Error converting lat/long:", lat_long_str)
    return None

# Build dictionary mapping state abbreviations -> list of regex patterns for
# matching variations of that state's name
def load_state_patterns(path):
    state_patterns = {}
    with open(path) as state_file:
        for line in state_file:
            bits = line.split("\t")
            abbrev = bits[0].strip()
            pattern_list = []
            for i in range(1, len(bits)):
                core = bits[i].strip()
                pattern_list.append(re.compile(r"\s+" + core + r"\s*$", re.I))
                pattern_list.append(re.compile(r"^" + core + r"\s*((,\s*)?" \
                                               + abbrev + r")?$", re.I))
            state_patterns[abbrev] = pattern_list
    return state_patterns

state_patterns = load_state_patterns("states.txt")

USA_Suffix = re.compile("([\\s*,]?\\s*USA?\\s*$)|([\\s*,]?\\s*united\\s*states\\s*(of\\s*america)?\\s*$)", re.I)

def normalize_location(location, timezone):
    if location:
        location = location.lower()
    match = USA_Suffix.search(location)
    stripped_loc = location
    if match:
        stripped_loc = location[:match.start()]
    if not stripped_loc:
        return None

    if location == "la":
        if not timezone:
            return None
        elif "pacific" in timezone:
            return "CA"
        elif "central" in timezone:
            return "LA"
        else:
            return None

    for abbrev, pattern_list in state_patterns.items():
        for pattern in pattern_list:
            if pattern.search(stripped_loc):
                return abbrev
    return None

def to_us_tz_abbrev(name):
    if not name:
        return None
    elif "/" not in name:
        state_name = name
    else:
        state_name = name.split("/")[1]

    for abbrev, pattern_list in state_patterns.items():
        for pattern in pattern_list:
            if pattern.search(state_name):
                return abbrev
    return None

def to_us_timezone(timezone_str):
    timezone = to_timezone(timezone_str)
    return to_us_tz_abbrev(timezone)

# Iterator that yields user_metadata, tweet_metadata, tweet with some cleanup.
def tweet_iter(input_file, count=-1):
    if count < 0:
        counter = itertools.repeat(None)
    else:
        counter = range(count*3)

    for _ in counter:
        user_metadata = get_all(input_file.readline().strip())
        tweet_metadata = input_file.readline().strip()
        tweet = input_file.readline()
        if tweet == "":
            print("EOF")
            return

        if len(user_metadata) <= User_Location_Col \
           or len(user_metadata) <= User_Timezone_Col:
            print("Fixed alignment issue")
            input_file.readline()
            input_file.readline()
            continue

        tweet_metadata = get_all(tweet_metadata)
        tweet = get(tweet.strip())
        #orig_copy = (str(user_metadata), str(tweet_metadata))

        # Cleanup user metadata
        if user_metadata[User_Timezone_Col]:
            user_metadata[User_Timezone_Col] = user_metadata[User_Timezone_Col].lower()

        if user_metadata[User_Location_Col]:
            user_metadata[User_Location_Col] = \
                normalize_location(user_metadata[User_Location_Col], \
                                   user_metadata[User_Timezone_Col])

        # Cleanup tweet metadata
        tweet_metadata[Post_Timezone_Col] = to_us_timezone(tweet_metadata[Post_Timezone_Col])
        if not user_metadata[User_Location_Col] and not tweet_metadata[Post_Timezone_Col]:
            continue

        try:
            time_str = tweet_metadata[Post_Time_Col]
            # Remove timezone code
            time_str = time_str[:-8] + time_str[-4:]
            tweet_metadata[Post_Time_Col] = \
                time.mktime(time.strptime(time_str, "%a %b %d %H:%M:%S %Y"))
        except ValueError as err:
            print(err)
            continue

        yield user_metadata, tweet_metadata, tweet#, orig_copy

def get_food_scores(filename):
    with open(filename) as score_file:
        healthy_foods = set()
        neutral_foods = set()
        unhealthy_foods = set()
        for line in score_file:
            tokens = line.strip().split("\t")
            if len(tokens) == 0:
                continue
            elif len(tokens) != 2:
                print("Error: line:", tokens)
                continue
            food, score = " ".join(tokenize(tokens[0])), int(tokens[1])
            if score == -1:
                healthy_foods.add(food)
            elif score == 0:
                neutral_foods.add(food)
            elif score == 1:
                unhealthy_foods.add(food)
            else:
                print("Error: Invalid score of", score, "for food", food)
        return healthy_foods, neutral_foods, unhealthy_foods

def for_each_post_in_timespans(timespans, on_post, on_period_end):
    timespan_iter = iter(timespans)
    post_count = 0
    with open("food_sample_2Oct2013_1Sep2021.txt") as big_file:
        timespan = next(timespan_iter)
        year, start_time, end_time, start_pos = timespan.get()
        big_file.seek(start_pos)
        try:
            for user_metadata, tweet_metadata, tweet in tweet_iter(big_file):
                if tweet_metadata[Post_Time_Col] < start_time:
                    continue
                elif tweet_metadata[Post_Time_Col] <= end_time:
                    on_post(timespan, user_metadata, tweet_metadata, tweet, big_file)
                else:
                    on_period_end(timespan)
                    timespan = next(timespan_iter)
                    year, start_time, end_time, start_pos = timespan.get()
                    big_file.seek(start_pos)
            on_period_end(timespan)
            print("Done")
        except StopIteration:
            on_period_end(timespan)
            print("Done")
        except Exception:
            print("Error at seek pos:", big_file.tell())
            print(traceback.format_exc())

def tokenize(tweet):
    tweet_doc = en_lang(tweet)
    #with tweet_doc.retokenize() as retokenizer:
    #    for i, token in enumerate(tweet_doc):
            # Make sure each hashtag is a single token
    #        if token.text == "#" and i < len(tweet_doc) - 1:
    #            retokenizer.merge(tweet_doc[i:i+2])

    lemmas = [term.lemma_.lower() for term in tweet_doc]
    return [term for term in lemmas \
            if not en_lang.vocab[term].is_stop \
            and not en_lang.vocab[term].is_punct]
