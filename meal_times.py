import parser, seasons
from datetime import datetime

curr_season = "Spring"

# Number of posts mentioning #breakfast
breakfast_count = 0
# Number of posts mentioning #lunch
lunch_count = 0
# Number of posts mentioning #dinner or #supper
dinner_count = 0

breakfast_average = 0
lunch_average = 0
dinner_average = 0

def to_minutes(timestamp):
    time = datetime.fromtimestamp(timestamp)
    if time.second >= 30:
        bonus = 1
    else:
        bonus = 0
    return time.hour * 60 + time.minute + bonus

def to_local_timezone(minutes, tz):
    print(tz)

def on_post(timespan, user_metadata, tweet_metadata, tweet, text_file):
    global breakfast_count, lunch_count, dinner_count
    time_in_minutes = to_minutes(tweet_metadata[parser.Post_Time_Col])
    time = to_local_timezone(time_in_minutes, tweet_metadata[parser.Post_Timezone_Col])
    tokens = parser.tokenize(tweet)
    if "#breakfast" in tokens:
        breakfast_count += 1
    if "#lunch" in tokens:
        lunch_count += 1
    if "#dinner" in tokens or "#supper" in tokens:
        dinner_count += 1

def on_period_end(timespan):
    global curr_season
    print("Finished processing", curr_season, timespan.year)

    curr_season = seasons.next_season(curr_season)

def main():
    parser.for_each_post_in_timespans(seasons.timespans, on_post, on_period_end)

main()
