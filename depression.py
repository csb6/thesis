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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
survey_period_batch_1 = [(datetime(2020, 5, 7),  datetime(2020, 5, 12)), \
                         (datetime(2020, 5, 14), datetime(2020, 5, 19)), \
                         (datetime(2020, 5, 21), datetime(2020, 5, 26)), \
                         (datetime(2020, 5, 28), datetime(2020, 6, 2)), \
                         (datetime(2020, 6, 4), datetime(2020, 6, 9)), \
                         (datetime(2020, 6, 11), datetime(2020, 6, 16)), \
                         (datetime(2020, 6, 18), datetime(2020, 6, 23)), \
                         (datetime(2020, 6, 25), datetime(2020, 6, 30)), \
                         (datetime(2020, 7, 2), datetime(2020, 7, 7)), \
                         (datetime(2020, 7, 9), datetime(2020, 7, 14)), \
                         (datetime(2020, 7, 16), datetime(2020, 7, 21))]

survey_period_batch_2 = [(datetime(2020, 8, 19), datetime(2020, 8, 31)), \
                         (datetime(2020, 9, 2), datetime(2020, 9, 14)), \
                         (datetime(2020, 9, 16), datetime(2020, 9, 28)), \
                         (datetime(2020, 9, 30), datetime(2020, 10, 12)), \
                         (datetime(2020, 10, 14),datetime(2020, 10, 26)), \
                         (datetime(2020, 10, 28),datetime(2020, 11, 9)), \
                         (datetime(2020, 11, 11),datetime(2020, 11, 23))]

survey_period_batch_3 = [(datetime(2021, 1, 6), datetime(2021, 1, 18)), \
                         (datetime(2021, 1, 20), datetime(2021, 2, 1)), \
                         (datetime(2021, 2, 3), datetime(2021, 2, 15)), \
                         (datetime(2021, 2, 17), datetime(2021, 3, 1)), \
                         (datetime(2021, 3, 3), datetime(2021, 3, 15)), \
                         (datetime(2021, 3, 17), datetime(2021, 3, 29))]

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
            state = statename_to_code[row["State"]]
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

depression_data = get_depression_data()
state_depression_scores = get_depression_scores_by_state(depression_data)
depression_medians_by_date = get_median_depression_scores_by_date(state_depression_scores)
state_classes_by_date = get_state_classes_by_date(state_depression_scores, \
                                                  depression_medians_by_date)

def read_tweets():
    while True:
        try:
            fields = [field.strip() for field in input().split("\t")]
        except EOFError:
            break
        if len(fields) != 4:
            print("Error parsing (wrong number of fields):", fields)
            continue
        state_code, _, post_time_str, tweet_text = fields
        if state_code == "":
            continue
        try:
            # Strip last 6 digits (hour, minute, second)
            post_date = datetime.strptime(post_time_str[:-6], "%Y%m%d")
        except ValueError:
            print(f"Failed to parse date string: '{post_time_str}'")
            continue
        yield state_code, post_date, tweet_text

# Creates map of form: day -> (state_code -> tweet_list)
def read_tweets_by_day_by_state():
    tweets_by_day_by_state = {}
    for state_code, post_date, tweet_text in read_tweets():
        if post_date not in tweets_by_day_by_state:
            tweets_by_day_by_state[post_date] = {}
        by_state = tweets_by_day_by_state[post_date]
        if state_code not in by_state:
            by_state[state_code] = []
        by_state[state_code].append(tweet_text)
    return tweets_by_day_by_state

# Replaces map with map of form: day -> (state_code -> [concatenated tweets])
def merge_tweets_by_day_by_state(tweets_by_day_by_state):
    for day, by_state in tweets_by_day_by_state.items():
        for state in by_state:
            by_state[state] = [" ".join(by_state[state])]

# Projects tweets_by_day_by_state to a table with three columns
# (docs, labels, state_labels), representing data that is within the range
# [start_date, end_date]
def flatten_to_columns(tweets_by_day_by_state, docs, labels, state_labels, \
                       start_date, end_date, period_state_labels):
    for day in tweets_by_day_by_state:
        if day < start_date or day > end_date:
            continue
        by_state = tweets_by_day_by_state[day]
        for state, tweets in by_state.items():
            label = period_state_labels[state]
            for tweet in tweets:
                docs.append(tweet)
                labels.append(label)
                state_labels.append(state)

# [X] 1. Need way to iterate over pairs of dates (periods)
# [ ] 2. Let P be the previous period. Let C be the current period.
#        Need to table-ify the tweets from P and feed them into a classifier
#        to predict the labels for C.
# [ ] 3. Repeat step 2 for each pair of adjacent periods in the test batch
# [ ] 4. Once model fully trained, iterate over pairs of periods (P, C) for
#        the dev batch, predict the labels of C using P
# [ ] 5. Tune the model based on the performance on the dev batch
# [ ] 6. Run over test data (some other batch of contiguous, same-length periods)
# [ ] 7. Repeat with different scikit models (see Thesis Log)

def for_each_adjacent_pair(period_list):
    for i in range(1, len(period_list)):
        prev_period = period_list[i-1]
        curr_period = period_list[i]
        yield prev_period, curr_period

def dummy(doc):
    return doc

def tokenize_on_spaces(doc):
    return doc.split(" ")

def convert_to_table(period, tweets_by_day_by_state, state_classes_by_date):
    docs = []
    labels = []
    state_labels = []
    start_date, end_date = period
    # TODO: have option to include multiple periods in the table
    flatten_to_columns(tweets_by_day_by_state, docs, labels, state_labels, \
                       start_date, end_date, \
                       state_classes_by_date[start_date])
    return docs, labels, state_labels

param_grid = [
    { 'C': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1, 10] }
]

param_grid_gb = [
    { "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.3], "max_depth": [1, 2, 3, 4] }
]

def classifier_1(prev_period, curr_period, tweets_by_day_by_state, state_classes_by_date):
    train_docs, train_labels, train_state_labels = convert_to_table(prev_period, \
                                                                    tweets_by_day_by_state, \
                                                                    state_classes_by_date)

    test_docs, test_labels, test_state_labels = convert_to_table(curr_period, \
                                                                 tweets_by_day_by_state, \
                                                                 state_classes_by_date)

    text_vectorizer = CountVectorizer(tokenizer=tokenize_on_spaces, preprocessor=dummy)
    X_train = text_vectorizer.fit_transform(train_docs)
    y_train = train_labels

    #classifier = GridSearchCV(LinearSVC(random_state=41, max_iter=2500), param_grid)
    #classifier = LinearSVC(random_state=41, max_iter=10000, C=0.1)
    #classifier = RandomForestClassifier(random_state=41)
    #classifier = LogisticRegression(random_state=41, max_iter=10000, C=0.1)
    classifier = GradientBoostingClassifier(random_state=41)
    classifier.fit(X_train, y_train)
    #print("Best params:", classifier.best_params_)

    X_test = text_vectorizer.transform(test_docs)
    y_test = test_labels
    y_pred = classifier.predict(X_test)
    return test_state_labels, y_test, y_pred

# Just use the previous period's classifications
def baseline_classifier(prev_period, curr_period, test_state_labels, \
                        state_classes_by_date, depression_medians_by_date):
    prev_start_date, prev_end_date = prev_period
    prev_labels_by_state = state_classes_by_date[prev_start_date]
    prev_median = depression_medians_by_date[prev_start_date]
    y_pred = []
    for state in test_state_labels:
        prev_label = prev_labels_by_state[state]
        y_pred.append(prev_label)
    return y_pred

def main():
    classifier_f1_sum = 0
    baseline_f1_sum = 0
    n = 0
    tweets_by_day_by_state = read_tweets_by_day_by_state()
    for prev_period, curr_period in for_each_adjacent_pair(survey_period_batch_1):
        test_state_labels, y_test, y_pred = classifier_1(prev_period, curr_period, \
                                                         tweets_by_day_by_state, \
                                                         state_classes_by_date)
        classifier_score = f1_score(y_test, y_pred)
        print("Classifier for", curr_period, ":", classifier_score)
        classifier_f1_sum += classifier_score
        n += 1

        baseline_y_pred = baseline_classifier(prev_period, curr_period, \
                                              test_state_labels, state_classes_by_date, \
                                              depression_medians_by_date)
        baseline_score = f1_score(y_test, baseline_y_pred)
        print("Baseline:", baseline_score)
        baseline_f1_sum += baseline_score

    print("Classifier: Average f1 score across all time periods:", classifier_f1_sum / n)
    print("Baseline: Average f1 score across all time periods:", baseline_f1_sum / n)

main()
#tweets_by_day_by_state = read_tweets_by_day_by_state()
#print(tweets_by_day_by_state)
#print()

#merge_tweets_by_day_by_state(tweets_by_day_by_state)
#print(tweets_by_day_by_state)
#print()

#docs = []
#labels = []
#state_labels = []
#start_date = datetime(2018, 3, 1, 0, 0)
#end_date = datetime(2018, 3, 7, 0, 0)
#period_state_labels = {"IL": 0, "NC": 1, "OH": 1, "CO": 0, "CA": 1}
#flatten_to_columns(tweets_by_day_by_state, docs, labels, state_labels, \
#                   start_date, end_date, period_state_labels)
#for doc, label, state_label in zip(docs, labels, state_labels):
#    print(doc, label, state_label, sep="|")
