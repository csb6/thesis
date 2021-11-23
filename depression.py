import random, datetime
import pandas as pd
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def get_depression_data():
    depression_data = pd.read_csv("depression_survey_data.csv", \
                                  parse_dates=["Time Period Start Date", \
                                               "Time Period End Date"],
                                  usecols=["Indicator", "State", "Group", \
                                           "Time Period Start Date", \
                                           "Time Period End Date", "Value"])
    return depression_data[(depression_data["Group"] == "By State") \
                           & (depression_data["Indicator"] == "Symptoms of Depressive Disorder")]

def tweets(data):
    for tweet in data:
        date_str, tweet_text = tweet.decode("utf-8").split("\n", 1)
        yield bytes(tweet_text, "utf-8")

def within_time_period(data, start_date, end_date):
    for tweet in data:
        date_str, tweet_text = tweet.decode("utf-8").split("\n", 1)
        date = datetime.datetime.fromisoformat(date_str)
        if start_date <= date and date <= end_date:
            yield bytes(tweet_text, "utf-8")

median = 28.9

def main():
    depression_data = get_depression_data()
    # start_date -> state -> score
    state_depression_scores = {}
    for timestamp, group in depression_data.groupby(["Time Period Start Date"]):
        for _, row in group.iterrows():
            state = row["State"]
            start_time = row["Time Period Start Date"]
            score = row["Value"]
            if start_time not in state_depression_scores:
                state_depression_scores[start_time] = {}
            assert state not in state_depression_scores[start_time]
            state_depression_scores[start_time][state] = score

    #print(state_depression_scores[pd.Timestamp('2021-08-18 00:00:00')])

    data = sklearn.datasets.load_files("tweets_by_state", random_state=45)
    depression_classes = []

    # The tweets
    X = within_time_period(data, datetime.datetime(2020, 4, 23), \
                           datetime.datetime(2020, 5, 5))
    scores_in_period = state_depression_scores[pd.Timestamp('2021-4-23 00:00:00')]
    y = []
    X_train, X_test = train_test_split(X)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #nb_classifier = MultinomialNM().fit(X_train_tfidf, )

    #depression_data[["Time Period Start Date", "Time Period End Date"]] \
    #    .diff(axis=1) \
    #    .join(depression_data[["Time Period Start Date", "Time Period End Date"]], \
    #          lsuffix='_caller', rsuffix='_other')

    #depression_by_date_and_class = classify_by_median(depression_by_date_and_state)

    #below_median, above_median = depression_by_date_and_class[datetime.datetime(2021, 9, 1)]
    #print("At/above median:", len(above_median))
    #print("Below median:", len(below_median))

    #print(depression_by_date_and_state[datetime.datetime(2021, 9, 1)])
    #print()
    #print("At/above median states/areas:", above_median)

main()
