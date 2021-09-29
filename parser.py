import time, os, re, itertools
from tzwhere import tzwhere

os.environ["TZ"] = "US/Pacific"
time.tzset()

tz = tzwhere.tzwhere()

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
    latitude, longitude = [float(n) for n in lat_long_str.split("|", 2)]
    return tz.tzNameAt(latitude, longitude)

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
            break

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
