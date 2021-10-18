import parser
from datetime import datetime

class TimePeriod:
    def __init__(self, start_date, end_date, start_pos):
        self.year = start_date.year
        self.start_date = start_date.timestamp()
        self.end_date = end_date.timestamp()
        self.start_pos = start_pos

    def get(self):
        return (self.year, self.start_date, self.end_date, self.start_pos)

# March 21
def spring(year, start_pos):
    return TimePeriod(datetime(year, 3, 21), datetime(year, 6, 20, 23, 59, 59), \
                      start_pos)

# June 21
def summer(year, start_pos):
    return TimePeriod(datetime(year, 6, 21), datetime(year, 9, 22, 23, 59, 59), \
                      start_pos)

# September 23
def fall(year, start_pos):
    return TimePeriod(datetime(year, 9, 23), datetime(year, 12, 21, 23, 59, 59), \
                      start_pos)

# December 22
def winter(year, start_pos):
    return TimePeriod(datetime(year, 12, 22), datetime(year+1, 3, 20, 23, 59, 59), \
                      start_pos)

             # 2015
timespans = [spring(2015, 2028808830), \
             summer(2015, 2415202330), \
             fall(2015, 2834021120), \
             winter(2015, 3213878123), \
             # 2016
             spring(2016, 3580091313), \
             summer(2016, 3956063644), \
             fall(2016, 4235801712), \
             winter(2016, 4546674691), \
             # 2017
             spring(2017, 4850694419), \
             summer(2017, 5119214979), \
             fall(2017, 5413981325), \
             winter(2017, 5662427657), \
             # 2018
             spring(2018, 6269206081), \
             summer(2018, 6520516548), \
             fall(2018, 6771411360), \
             winter(2018, 6994153592), \
             # 2019
             spring(2019, 7210788125), \
             summer(2019, 7382373769), \
             fall(2019, 7548119129), \
             winter(2019, 7693301002), \
             # 2020
             spring(2020, 7857250194), \
             summer(2020, 7997432869), \
             fall(2020, 8133269478), \
             winter(2020, 8271489464), \
             # 2021
             spring(2021, 8376121327), \
             summer(2021, 8481089485)]

curr_season = "Spring"

# Total number of localized U.S. posts in current period that contain
# #breakfast, #lunch, #dinner, or #supper
post_count = 0

# Number of posts mentioning #breakfast
breakfast_count = 0
# Number of posts mentioning #brunch
brunch_count = 0
# Number of posts mentioning #lunch
lunch_count = 0
# Number of posts mentioning #dinner or #supper
dinner_count = 0

# Cross-mentioning posts
breakfast_and_brunch_count = 0
breakfast_and_lunch_count = 0
breakfast_and_dinner_count = 0
brunch_and_lunch_count = 0
brunch_and_dinner_count = 0
lunch_and_dinner_count = 0

def on_post(timespan, user_metadata, tweet_metadata, tweet, text_file):
    global post_count, breakfast_count, brunch_count, lunch_count, \
        dinner_count, breakfast_and_brunch_count, breakfast_and_lunch_count, \
        breakfast_and_dinner_count, brunch_and_lunch_count, \
        brunch_and_dinner_count, lunch_and_dinner_count
    is_match = False
    is_breakfast = False
    is_brunch = False
    is_lunch = False
    is_dinner = False

    tokens = parser.tokenize(tweet)
    if "#breakfast" in tokens:
        breakfast_count += 1
        is_breakfast = True
        is_match = True
    if "#brunch" in tokens:
        brunch_count += 1
        is_brunch = True
        is_match = True
    if "#lunch" in tokens:
        lunch_count += 1
        is_lunch = True
        is_match = True
    if "#dinner" in tokens or "#supper" in tokens:
        dinner_count += 1
        is_dinner = True
        is_match = True

    if not is_match:
        return

    if is_breakfast and is_brunch:
        breakfast_and_brunch_count += 1
    if is_breakfast and is_lunch:
        breakfast_and_lunch_count += 1
    if is_breakfast and is_dinner:
        breakfast_and_dinner_count += 1
    if is_brunch and is_lunch:
        brunch_and_lunch_count += 1
    if is_brunch and is_dinner:
        brunch_and_dinner_count += 1
    if is_lunch and is_dinner:
        lunch_and_dinner_count += 1

    post_count += 1

def report(name, value):
    print(" ", name, value, \
          "({:.2f}% of posts with meal matches in period)".format(value / post_count * 100))

def on_period_end(timespan):
    global curr_season, post_count, breakfast_count, brunch_count, lunch_count, \
        dinner_count, breakfast_and_brunch_count, breakfast_and_lunch_count, \
        breakfast_and_dinner_count, brunch_and_lunch_count, \
        brunch_and_dinner_count, lunch_and_dinner_count
    print("Finished processing", curr_season, timespan.year)
    print(" Post count:", post_count)
    report("#breakfast", breakfast_count)
    report("#brunch", brunch_count)
    report("#lunch", lunch_count)
    report("#dinner/#supper", dinner_count)
    print()
    print(" Combinations:")
    report("#breakfast and #brunch", breakfast_and_brunch_count)
    report("#breakfast and #lunch", breakfast_and_lunch_count)
    report("#breakfast and #dinner", breakfast_and_dinner_count)
    report("#brunch and #lunch", brunch_and_lunch_count)
    report("#brunch and #dinner", brunch_and_dinner_count)
    report("#lunch and #dinner", lunch_and_dinner_count)

    # Reset everything
    post_count = 0
    breakfast_count = 0
    brunch_count = 0
    lunch_count = 0
    dinner_count = 0
    breakfast_and_brunch_counts = 0
    breakfast_and_lunch_counts = 0
    breakfast_and_dinner_counts = 0
    brunch_and_lunch_counts = 0
    brunch_and_dinner_counts = 0
    lunch_and_dinner_counts = 0

    if curr_season == "Spring":
        curr_season = "Summer"
    elif curr_season == "Summer":
        curr_season = "Fall"
    elif curr_season == "Fall":
        curr_season = "Winter"
    elif curr_season == "Winter":
        curr_season = "Spring"

def main():
    parser.for_each_post_in_timespans(timespans, on_post, on_period_end)

main()
