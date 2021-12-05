import parser, twokenize, os, os.path
from datetime import datetime

def main():
    cached_state_dirs = set()
    cached_year_dirs = set()
    with open("food_sample_2Oct2013_1Sep2021.txt") as big_file:
        for user_meta, tweet_meta, tweet in parser.tweet_iter(big_file):
            state_code = user_meta[parser.User_Location_Col]
            if not state_code:
                continue
            post_time = datetime.fromtimestamp(tweet_meta[parser.Post_Time_Col])
            year_dir = post_time.year
            tweet_text = " ".join(twokenize.tokenizeRawTweetText(tweet))

            if state_code not in cached_state_dirs:
                os.mkdir(f"tweets_by_day/{state_code}")
                cached_state_dirs.add(state_code)
            if (state_code, year_dir) not in cached_year_dirs:
                os.mkdir(f"tweets_by_day/{state_code}/{year_dir}")
                cached_year_dirs.add((state_code, year_dir))
            day_file = f"{post_time.month}-{post_time.day}.txt"
            with open(f"tweets_by_day/{state_code}/{year_dir}/{day_file}", "a+") as f:
                f.write(tweet_text + "\n")

main()
