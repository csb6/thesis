import parser, random

def print_user_metadata(metadata):
    print("Username:", metadata[parser.Username_Col])
    print("Name:", metadata[parser.Name_Col])
    print("User Location:", metadata[parser.User_Location_Col])
    print("User Timezone:", metadata[parser.User_Timezone_Col])

def print_tweet_metadata(metadata):
    print("Post time:", metadata[parser.Post_Time_Col])
    print("Post timezone:", metadata[parser.Post_Timezone_Col])

start_pos = 7848556306 # 1st post of Mar 15, 2020

def main():
    with open("food_sample_2Oct2013_1Sep2021.txt") as big_file:
        big_file.seek(start_pos)

        for user_metadata, tweet_metadata, tweet in parser.tweet_iter(big_file):
            if random.randint(1, 1000) == 50:
                print(user_metadata)
                print(tweet_metadata)
                print()

main()
