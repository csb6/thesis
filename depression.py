import random, re, statistics, os, os.path
from datetime import datetime, date, timedelta
import pandas as pd
import sklearn.datasets, numpy
import parser, twokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

punct = twokenize.regex_or(twokenize.punctSeq, r"[\(\)\[\]\*\-\+\/\—@]", r"RT")
url_or_punct = re.compile(twokenize.regex_or(twokenize.url, punct))

statename_to_code = {"Alabama": "AL", \
                     "Alaska": "AK", \
                     "Arizona": "AZ", \
                     "Arkansas": "AR", \
                     'California': "CA", \
                     'Colorado': "CO", \
                     'Connecticut': "CT", \
                     'Delaware': "DE", \
                     'District of Columbia': "DC", \
                     'Florida': "FL", \
                     'Georgia': "GA", \
                     'Hawaii': "HI", \
                     'Idaho': "ID", \
                     'Illinois': "IL", \
                     'Indiana': "IN", \
                     'Iowa': "IA", \
                     'Kansas': "KS", \
                     'Kentucky': "KY", \
                     'Louisiana': "LA", \
                     'Maine': "ME", \
                     'Maryland': "MD", \
                     'Massachusetts': "MA", \
                     'Michigan': "MI", \
                     'Minnesota': "MN", \
                     'Mississippi': "MS", \
                     'Missouri': "MO", \
                     'Montana': "MT", \
                     'Nebraska': "NE", \
                     'Nevada': "NV", \
                     'New Hampshire' : "NH", \
                     'New Jersey': "NJ", \
                     'New Mexico': "NM", \
                     'New York': "NY", \
                     'North Carolina': "NC", \
                     'North Dakota': "ND", \
                     'Ohio': "OH", \
                     'Oklahoma': "OK", \
                     'Oregon': "OR", \
                     'Pennsylvania': "PA", \
                     'Rhode Island': "RI", \
                     'South Carolina': "SC", \
                     'South Dakota': "SD", \
                     'Tennessee': "TN", \
                     'Texas': "TX", \
                     'Utah': "UT", \
                     'Vermont': "VT", \
                     'Virginia': "VA", \
                     'Washington': "WA", \
                     'West Virginia': "WV", \
                     'Wisconsin': "WI", \
                     'Wyoming': "WY"}

# Load a CSV containing depression survey scores by date by state
def get_depression_data():
    depression_data = pd.read_csv("depression_survey_data.csv", \
                                  parse_dates=["Time Period Start Date", \
                                               "Time Period End Date"],
                                  usecols=["Indicator", "State", "Group", \
                                           "Time Period Start Date", \
                                           "Time Period End Date", "Value"])
    return depression_data[(depression_data["Group"] == "By State") \
                           & (depression_data["Indicator"] == "Symptoms of Depressive Disorder")]

# Given depression survey scores by date by state, return a map from
# start dates -> state -> depression score
def get_depression_scores_by_state(depression_data):
    # start_date -> state -> depression score
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
    return state_depression_scores

def get_median_depression_scores_by_date(state_depression_scores):
    # start_date -> median depression score
    depression_medians_by_date = {}
    for start_date, states_data in state_depression_scores.items():
        depression_medians_by_date[start_date] = \
            statistics.median(state_depression_scores[start_date].values())
    return depression_medians_by_date

def get_tweet_filenames_and_dates(state_code, start_date, end_date):
    assert start_date.year == end_date.year
    root = f"tweets_by_day/{state_code}/{start_date.year}"
    filenames_and_dates = []
    curr_date = start_date
    while curr_date <= end_date:
        filename = f"{root}/{curr_date.month}-{curr_date.day}.txt"
        if os.path.isfile(filename):
            filenames_and_dates.append((filename, pd.Timestamp(curr_date)))
        curr_date += timedelta(days=1)
    return filenames_and_dates

def add_tweets(tweets, labels, label, start_date, end_date, \
               curr_state, curr_state_code):
    filenames_and_dates = get_tweet_filenames_and_dates(curr_state_code, \
                                                        start_date, end_date)
    for filename, date in filenames_and_dates:
        with open(filename) as day_file:
            for line in day_file:
                tokens = [t.lower() for t in line.strip().split() if not re.match(url_or_punct, t)]
                tweets.append(tokens)
                labels.append(label)

def dummy(doc):
    return doc

def main():
    print("Loading depression survey data...")
    depression_data = get_depression_data()
    state_depression_scores = get_depression_scores_by_state(depression_data)
    depression_medians_by_date = get_median_depression_scores_by_date(state_depression_scores)

    print("Loading tweet data...")
    tweets = []
    labels = []

    start_date = datetime(2020, 4, 23)
    end_date = datetime(2020, 5, 5)
    median_for_period = depression_medians_by_date[start_date]
    for curr_state, curr_state_code in statename_to_code.items():
        score = state_depression_scores[start_date][curr_state]
        if score > median_for_period:
            label = 1 # Above median
        else:
            label = 0 # At/below median
        print("Label for", curr_state, "is", label)
        add_tweets(tweets, labels, label, start_date, end_date, curr_state, curr_state_code)

    print("Calculating tf-idf scores...")
    tfidf_vectorizer = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy)
    X = tfidf_vectorizer.fit_transform(tweets)

    print("Partitioning into train/test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state=40)

    print("Training classifier...")
    classifier = LinearSVR(random_state=41).fit(X_train, y_train)
    X_train_pred = numpy.rint(classifier.predict(X_test))
    print("Accuracy:", accuracy_score(X_train_pred, y_test))

main()
