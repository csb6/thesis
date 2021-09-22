import parser

def print_user_metadata(metadata):
    print("Username:", metadata[parser.Username_Col])
    print("Name:", metadata[parser.Name_Col])
    print("User Location:", metadata[parser.User_Location_Col])
    print("User Timezone:", metadata[parser.User_Timezone_Col])

def print_tweet_metadata(metadata):
    print("Post time:", metadata[parser.Post_Time_Col])
    print("Post timezone:", metadata[parser.Post_Timezone_Col])

def print_all(reader):
    for user_metadata, tweet_metadata, tweet in parser.tweet_iter(reader, 100):
        print_user_metadata(user_metadata)
        print_tweet_metadata(tweet_metadata)
        print(tweet)
        print()


def main():
    with open("food_sample_2Oct2013_1Sep2021.txt") as big_file:
        print_all(big_file)

main()
