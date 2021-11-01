from parser import *

def main():
    user_fields = [Username_Col, Name_Col, User_ID_Col, User_Location_Col, \
                   User_Timezone_Col, User_Lang_Col]
    post_fields = [Post_Time_Col, Post_Timezone_Col]
    with open("food_sample_2Oct2013_1Sep2021.txt") as big_file:
        for user_metadata, tweet_metadata, tweet in tweet_iter(big_file):
            print("\t".join([str(user_metadata[field]) for field in user_fields]))
            print("\t".join([str(tweet_metadata[field]) for field in post_fields]))
            print(tweet)

main()
