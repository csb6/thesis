import parser, twokenize, re
from datetime import datetime

twitter_url = re.compile("^(http|https)://t\.co")
total_tweets = 0
total_tweets_with_trailing_urls = 0
total_by_year = {}
total_trailing_by_year = {}

def print_result(total, total_with_urls):
    print("Total number of tweets:", total)
    print("Total number of tweets with Twitter URLs at end:", \
          total_with_urls)
    print("Percentage with Twitter URLs at end:", \
          total_with_urls / total)
    print()

def main():
    global total_tweets, total_tweets_with_trailing_urls
    with open("food_sample_2Oct2013_1Sep2021.txt") as big_file:
        for user_meta, tweet_meta, tweet in parser.tweet_iter(big_file):
            if not tweet_meta[parser.Post_Time_Col]:
                continue
            tokens = tweet.strip().split()
            if not tokens:
                continue
            year = datetime.fromtimestamp(tweet_meta[parser.Post_Time_Col]).year
            if year not in total_by_year:
                total_by_year[year] = 0
            if year not in total_trailing_by_year:
                total_trailing_by_year[year] = 0
            total_tweets += 1
            total_by_year[year] += 1
            url = tokens[-1].strip()
            if twitter_url.match(url):
                total_tweets_with_trailing_urls += 1
                total_trailing_by_year[year] += 1
    print_result(total_tweets, total_tweets_with_trailing_urls)

    print("By year:\n")
    for year, total_with_url in total_trailing_by_year.items():
        year_total = total_by_year[year]
        print_result(year_total, total_with_url)

main()
