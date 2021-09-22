import parser, datetime, traceback

timespans = []

start_time = datetime.datetime(2020, 3, 15).timestamp()
end_time = datetime.datetime(2020, 5, 15).timestamp()

start_pos = 7848556306 # 1st post of Mar 15, 2020
#start_pos = 4302049791
#start_pos = 0

def main():
    total_post_count = 0
    post_count = 0
    first_post = 0
    uniq_user_locs = set()
    uniq_tz = set()
    with open("food_sample_2Oct2013_1Sep2021.txt") as big_file:
        big_file.seek(start_pos)
        big_file.tell()
        try:
            for user_metadata, tweet_metadata, tweet in parser.tweet_iter(big_file):
                total_post_count += 1
                if tweet_metadata[parser.Post_Time_Col] < start_time:
                    continue
                elif tweet_metadata[parser.Post_Time_Col] <= end_time:
                    if user_metadata[parser.User_Location_Col]:
                        uniq_user_locs.add(user_metadata[parser.User_Location_Col])
                    if tweet_metadata[parser.Post_Timezone_Col]:
                        uniq_tz.add(tweet_metadata[parser.Post_Timezone_Col])
                    post_count += 1
                    if post_count == 1:
                        print("Seek point for first post:", big_file.tell())
                else:
                    print(end_time, tweet_metadata[parser.Post_Time_Col])
                    break
        except StopIteration:
            print("EOF")
        except Exception as err:
            print("Error at seek pos:", big_file.tell(), "and tweet #", total_post_count)
            print(traceback.format_exc())
    print("Post count (in time period):", post_count)
    print("Total post count", total_post_count)
    print("Locations", sorted(list(uniq_user_locs)))
    print("Timezones", sorted(list(uniq_tz)))

main()
