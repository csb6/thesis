import gzip, io, time, os, re
from tzwhere import tzwhere

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

USA_Suffix = re.compile("([\\s*,]?\\s*USA?\\s*$)|([\\s*,]?\\s*united\\s*states\\s*(of\\s*america)?\\s*$)", re.I)

def normalize_location(location, timezone, states):
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

    for abbrev, state_patterns in states.items():
        for pattern in state_patterns:
            if pattern.search(stripped_loc):
                return abbrev
    return None

state_patterns = load_state_patterns("states.txt")

# Iterator that yields user_metadata, tweet_metadata, tweet with some cleanup.
def entry_iter(input_file, count):
    reader = iter(input_file)
    try:
        for i in range(count*3):
            user_metadata = get_all(next(reader))
            tweet_metadata = get_all(next(reader))
            tweet = get(next(reader))

            # Cleanup user metadata
            if not user_metadata[User_Location_Col] or not user_metadata[User_Timezone_Col]:
                continue
            user_metadata[User_Location_Col] = user_metadata[User_Location_Col].lower()
            user_metadata[User_Timezone_Col] = user_metadata[User_Timezone_Col].lower()
            user_metadata[User_Location_Col] = \
                normalize_location(user_metadata[User_Location_Col], \
                                   user_metadata[User_Timezone_Col], state_patterns)

            # Cleanup tweet metadata
            tweet_metadata[Post_Timezone_Col] = to_timezone(tweet_metadata[Post_Timezone_Col])
            try:
                time_str = tweet_metadata[Post_Time_Col]
                tweet_metadata[Post_Time_Col] = \
                    time.mktime(time.strptime(time_str, "%a %b %d %H:%M:%S %Z %Y"))
            except ValueError as err:
                print(err)
                continue

            yield user_metadata, tweet_metadata, tweet
    except StopIteration:
        pass

def print_user_metadata(metadata):
    print("Username:", metadata[Username_Col])
    print("Name:", metadata[Name_Col])
    print("User Location:", metadata[User_Location_Col])
    print("User Timezone:", metadata[User_Timezone_Col])

def print_tweet_metadata(metadata):
    print("Post time:", metadata[Post_Time_Col])
    print("Post timezone:", metadata[Post_Timezone_Col])

def print_all(reader):
    for user_metadata, tweet_metadata, tweet in entry_iter(reader, 100):
        print_user_metadata(user_metadata)
        print_tweet_metadata(tweet_metadata)
        print(tweet)
        print()

def open_text_gzip(filename):
    bin_file = gzip.open(filename)
    return io.TextIOWrapper(bin_file, encoding="utf-8")

def main():
    os.environ["TZ"] = "US/Pacific"
    time.tzset()
    with open_text_gzip("food_sample_2Oct2013_1Sep2021.txt.gz") as big_file:
        print_all(big_file)

main()
