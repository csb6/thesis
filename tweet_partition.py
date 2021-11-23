import random, parser, os, os.path, datetime

rand_gen = random.Random()
rand_gen.seed(45)

partitions = ["train", "dev", "test"]
weights = [0.6, 0.2, 0.2]

def get_partition():
    return rand_gen.choices(partitions, weights)

def main():
    state_indices = {}
    with open("food_sample_2Oct2013_1Sep2021.txt") as big_file:
        for user_meta, tweet_meta, tweet in parser.tweet_iter(big_file):
            if not user_meta[parser.User_Location_Col]:
                continue
            state_code = user_meta[parser.User_Location_Col]
            tweet_dir = "tweets_by_state/{}".format(state_code)
            if not os.path.isdir(tweet_dir):
                os.mkdir(tweet_dir)
                state_indices[state_code] = 0
            with open(tweet_dir + "/" + str(state_indices[state_code]), "w") as tweet_file:
                time = datetime.datetime.fromtimestamp(tweet_meta[parser.Post_Time_Col])
                tweet_file.write(f"{time}\n")
                tweet_file.write(tweet + "\n")
                state_indices[state_code] += 1

main()
