import parser, seasons

curr_season = "Spring"

# Total number of localized U.S. posts
total_post_count = 0
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
        brunch_and_dinner_count, lunch_and_dinner_count, total_post_count
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
    total_post_count += 1

def report(name, value):
    print(" ", name, value, \
          "({:.2f}% of posts with meal matches in period)".format(value / post_count * 100))

def on_period_end(timespan):
    global curr_season, post_count, breakfast_count, brunch_count, lunch_count, \
        dinner_count, breakfast_and_brunch_count, breakfast_and_lunch_count, \
        breakfast_and_dinner_count, brunch_and_lunch_count, \
        brunch_and_dinner_count, lunch_and_dinner_count, total_post_count
    print("Finished processing", curr_season, timespan.year)
    print(" Total localized posts:", total_post_count)
    print(" Post count (meal hashtags):", post_count)
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
    total_post_count = 0
    post_count = 0
    breakfast_count = 0
    brunch_count = 0
    lunch_count = 0
    dinner_count = 0
    breakfast_and_brunch_count = 0
    breakfast_and_lunch_count = 0
    breakfast_and_dinner_count = 0
    brunch_and_lunch_count = 0
    brunch_and_dinner_count = 0
    lunch_and_dinner_count = 0

    curr_season = seasons.next_season(curr_season)

def main():
    parser.for_each_post_in_timespans(seasons.timespans, on_post, on_period_end)

main()
