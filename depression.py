import random, re, statistics, os, os.path, copy
from datetime import datetime, date, timedelta
import pandas as pd
import sklearn.datasets, numpy
import twokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, \
    CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, \
    GridSearchCV
from sklearn.metrics import classification_report, precision_score, \
    recall_score, accuracy_score, f1_score

# Tokens to remove from the tweets in the dataset
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

# The time periods in which the CDC depression data provides statewide
# depression rates for the survey
survey_periods = [(datetime(2020, 4, 23), datetime(2020, 5, 5)), \
                  (datetime(2020, 5, 7),  datetime(2020, 5, 12)), \
                  (datetime(2020, 5, 14), datetime(2020, 5, 19)), \
                  (datetime(2020, 5, 21), datetime(2020, 5, 26)), \
                  (datetime(2020, 5, 28), datetime(2020, 6, 2)), \
                  (datetime(2020, 6, 4), datetime(2020, 6, 9)), \
                  (datetime(2020, 6, 11), datetime(2020, 6, 16)), \
                  (datetime(2020, 6, 18), datetime(2020, 6, 23)), \
                  (datetime(2020, 6, 25), datetime(2020, 6, 30)), \
                  (datetime(2020, 7, 2), datetime(2020, 7, 7)), \
                  (datetime(2020, 7, 9), datetime(2020, 7, 14)), \
                  (datetime(2020, 7, 16), datetime(2020, 7, 21)), \
                  (datetime(2020, 8, 19), datetime(2020, 8, 31)), \
                  (datetime(2020, 9, 2), datetime(2020, 9, 14)), \
                  (datetime(2020, 9, 16), datetime(2020, 9, 28)), \
                  (datetime(2020, 9, 30), datetime(2020, 10, 12)), \
                  (datetime(2020, 10, 14),datetime(2020, 10, 26)), \
                  (datetime(2020, 10, 28),datetime(2020, 11, 9)), \
                  (datetime(2020, 11, 11),datetime(2020, 11, 23)), \
                  (datetime(2020, 11, 25),datetime(2020, 12, 7)), \
                  (datetime(2020, 12, 9),datetime(2020, 12, 21)), \
                  (datetime(2021, 1, 6), datetime(2021, 1, 18)), \
                  (datetime(2021, 1, 20), datetime(2021, 2, 1)), \
                  (datetime(2021, 2, 3), datetime(2021, 2, 15)), \
                  (datetime(2021, 2, 17), datetime(2021, 3, 1)), \
                  (datetime(2021, 3, 3), datetime(2021, 3, 15)), \
                  (datetime(2021, 3, 17), datetime(2021, 3, 29)), \
                  (datetime(2021, 4, 14), datetime(2021, 4, 26)), \
                  (datetime(2021, 4, 28), datetime(2021, 5, 10)), \
                  (datetime(2021, 5, 12), datetime(2021, 5, 24)), \
                  (datetime(2021, 5, 26), datetime(2021, 6, 7)), \
                  (datetime(2021, 6, 9), datetime(2021, 6, 21)), \
                  (datetime(2021, 6, 23), datetime(2021, 7, 5)), \
                  (datetime(2021, 7, 21), datetime(2021, 8, 2)), \
                  (datetime(2021, 8, 4), datetime(2021, 8, 16)), \
                  (datetime(2021, 8, 18), datetime(2021, 8, 30)), \
                  (datetime(2021, 9, 1), datetime(2021, 9, 13))]

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

# Create a map from the start dates of each period to the median score
# for that period.
def get_median_depression_scores_by_date(state_depression_scores):
    # start_date -> median depression score
    depression_medians_by_date = {}
    for start_date, states_data in state_depression_scores.items():
        depression_medians_by_date[start_date] = \
            statistics.median(state_depression_scores[start_date].values())
    return depression_medians_by_date

def get_state_classes_by_date(state_depression_scores, depression_medians_by_date):
    # start_date -> state -> class
    state_classes_by_date = {}
    for start_date, state_scores in state_depression_scores.items():
        median = depression_medians_by_date[start_date]
        assert start_date not in state_classes_by_date
        state_classes_by_date[start_date] = {}
        for state, score in state_scores.items():
            assert state not in state_classes_by_date[start_date]
            if score > median:
                state_classes_by_date[start_date][state] = 1
            else:
                state_classes_by_date[start_date][state] = 0
    return state_classes_by_date

def get_tweet_filenames_and_dates(state_code, start_date, end_date):
    assert start_date.year == end_date.year
    root = f"tweets_by_day/{state_code}/{start_date.year}"
    filenames_and_dates = []
    curr_date = start_date
    while curr_date <= end_date:
        filename = f"{root}/{curr_date.month}-{curr_date.day}.txt"
        if os.path.isfile(filename):
            filenames_and_dates.append((filename, pd.Timestamp(copy.deepcopy(curr_date))))
        curr_date += timedelta(days=1)
    return filenames_and_dates


depression_data = get_depression_data()
state_depression_scores = get_depression_scores_by_state(depression_data)
depression_medians_by_date = get_median_depression_scores_by_date(state_depression_scores)
state_classes_by_date = get_state_classes_by_date(state_depression_scores, \
                                                  depression_medians_by_date)

file_cache = {}

# Add each tweet from the time period start_date-end_date as separate documents.
# docs, labels, state_labels, and date_labels are all equal length lists and will
# be updated together. This is used as the training data.
def add_tweets(docs, labels, statename_and_codes, median_for_period, start_date, \
               end_date, state_labels=None, date_labels=None):
    for curr_state, curr_state_code in statename_and_codes:
        # Determine the depression score and class of each state in this
        # time period
        label = state_classes_by_date[start_date][curr_state]

        # Create data entries for each tokenized tweet, associating with the class
        # given to its state
        filenames_and_dates = get_tweet_filenames_and_dates(curr_state_code, \
                                                            start_date, end_date)

        for filename, date in filenames_and_dates:
            if filename in file_cache:
                tweets = file_cache[filename]
                for tweet in tweets:
                    docs.append(tweet)
                    labels.append(label)
                    if state_labels is not None:
                        state_labels.append(curr_state)
                    if date_labels is not None:
                        date_labels.append(start_date)
            else:
                tweets = []
                with open(filename) as day_file:
                    for line in day_file:
                        tweet = []
                        for token in line.strip().split():
                            if re.match(url_or_punct, token) or token in ENGLISH_STOP_WORDS:
                                continue
                            tweet.append(token)
                        docs.append(tweet)
                        labels.append(label)
                        if state_labels is not None:
                            state_labels.append(curr_state)
                        if date_labels is not None:
                            date_labels.append(start_date)
                        tweets.append(tweet)
                file_cache[filename] = tweets

file_cache2 = {}

# Similar to add_tweets, but concatenates all tweets from each state on a given day
# into individual documents. This is used to as the test data.
def add_day_tweets(docs, labels, statename_and_codes, median_for_period, start_date, \
                   end_date, state_labels=None, date_labels=None):
    for curr_state, curr_state_code in statename_and_codes:
        # Determine the depression score and class of each state in this
        # time period
        label = state_classes_by_date[start_date][curr_state]

        # Create data entries for each tokenized tweet, associating with the class
        # given to its state
        filenames_and_dates = get_tweet_filenames_and_dates(curr_state_code, \
                                                            start_date, end_date)

        for filename, date in filenames_and_dates:
            if filename in file_cache2:
                tokens = file_cache2[filename]
            else:
                tokens = []
                with open(filename) as day_file:
                    for line in day_file:
                        for token in line.strip().split():
                            if re.match(url_or_punct, token) or token in ENGLISH_STOP_WORDS:
                                continue
                            tokens.append(token)
                file_cache2[filename] = tokens
            docs.append(tokens)
            labels.append(label)
            if state_labels is not None:
                state_labels.append(curr_state)
            if date_labels is not None:
                date_labels.append(start_date)

def dummy(doc):
    return doc

param_grid = [
    { 'C': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1, 10] }
]

# Just use the previous period's score classifications
def baseline_classifier(prev_state_depression_scores, prev_median, state_labels):
    y_pred = []
    for i in range(len(state_labels)):
        curr_state = state_labels[i]
        prev_score = prev_state_depression_scores[curr_state]
        if prev_score > prev_median:
            label = 1
        else:
            label = 0
        y_pred.append(label)
    return y_pred

def main_classifier(i, next_start_date, next_end_date):
    median_for_next_period = depression_medians_by_date[next_start_date]
    docs = []
    labels = []
    date_labels = []
    state_labels = []

    for start_date, end_date in survey_periods[:i]:
        median_for_period = depression_medians_by_date[start_date]
        add_tweets(docs, labels, statename_to_code.items(), \
                   median_for_period, start_date, end_date, \
                   state_labels, date_labels)

    test_docs = []
    test_labels = []
    test_state_labels = []
    test_date_labels = []
    add_day_tweets(test_docs, test_labels, statename_to_code.items(), \
                   median_for_next_period, next_start_date, next_end_date, \
                   test_state_labels, test_date_labels)

    text_vectorizer = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy)
    X_train = text_vectorizer.fit_transform(docs)
    y_train = labels
    X_test = text_vectorizer.transform(test_docs)
    y_test = test_labels

    #X_train1, X_dev, y_train1, y_dev = train_test_split(X_train, y_train, random_state=40)
    #grid_search = GridSearchCV(LinearSVC(random_state=41, max_iter=2000), param_grid)
    #grid_search.fit(X_train1, y_train1)
    #print("Best params:", grid_search.best_params_)
    #continue
    classifier = LinearSVC(random_state=41, max_iter=2500, C=0.1).fit(X_train, y_train)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    return y_test, y_pred

def main():
    f1_sum = 0
    n = 0
    for i in range(1, len(survey_periods)):
        next_start_date, next_end_date = survey_periods[i]
        print("Predicting for:", next_start_date, "thru", next_end_date)
        y_test, y_pred = main_classifier(i, next_start_date, next_end_date)

        score = f1_score(y_test, y_pred)
        print(score)
        f1_sum += score
        n += 1

    print("Average f1 score across all time periods:", f1_sum / n)

main()
