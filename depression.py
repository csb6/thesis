import random, re, statistics, os, os.path, copy
from datetime import datetime, date, timedelta
from statistics import mode
import pandas as pd
import sklearn.datasets, numpy
import twokenize
#import parser
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
from sklearn.impute import SimpleImputer

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

print(len(survey_periods))

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
#healthy_foods, neutral_foods, unhealthy_foods = parser.get_food_scores("food_scores.txt")

def print_depression_data():
    print("start-date CA TX NY AZ median")
    for start_date, states_data in state_depression_scores.items():
        print(start_date.strftime("%Y-%m-%d"), states_data["CA"], states_data["TX"], states_data["NY"], \
              states_data["AZ"], depression_medians_by_date[start_date])

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

def dummy(doc):
    return doc

def tokenize_on_spaces(doc):
    return [token.strip(twokenize.punctChars) for token in doc.split(" ") \
            if not re.match(url_or_punct, token)]

def tokenize_food_words(doc):
    return [token for token in doc.split(" ") \
            if token in healthy_foods or token in neutral_foods \
            or token in unhealthy_foods]

def tokenize_hashtags(doc):
    return [token for token in doc.split(" ") if token.startswith("#")]

param_grid = [
    { 'C': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1, 10] }
]

param_grid_rf = [
    {'n_estimators': [100, 200, 500]}
]

param_grid_gb = [
    { "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.3] }
]

# TODO:
#  - Eliminate monolithic versions of run_time_series and run_tagged_words
#  - Verify the modular versions work the same!
#  - Implement combo classifier by merging the feature arrays, etc.
#  - Try filtering to only food words, hastags, etc. (see thesis log suggestions)
def run_combo_classifier(classifier, k, periods,
                         tweets_by_day_by_state, state_classes_by_date,
                         depression_medians_by_date, use_mode=False):
    pass

def run_time_series_classifier(training_periods):
    states_to_time_series = {}
    for state in statename_to_code.values():
        classes = []
        for date, _ in training_periods:
            state_class = state_classes_by_date[date][state]
            median = depression_medians_by_date[date]
            if state_class == 0:
                assert(state_depression_scores[date][state] <= median)
            else:
                assert(state_depression_scores[date][state] > median)
            classes.append(state_class)
        states_to_time_series[state] = classes

    k = 3
    time_series_f1_sum = 0
    n = 0
    for j in range(k+1, len(training_periods)):
        X_train = []
        y_train = []
        # Training data: every len(k) time series from every state in period range [k, -j)
        #   Grouped by period
        i = 0
        while i+k < j:
            for state, series in states_to_time_series.items():
                X_train.append(series[i:i+k])
                y_train.append(series[i+k])
            i += 1

        X_test = []
        y_test = []
        # Test data: every len(k) time series from every state for period j
        for state, series in states_to_time_series.items():
            X_test.append(series[j-k:j])
            y_test.append(series[j])

        classifier = LinearSVC(random_state=41, max_iter=10000, C=0.01)
        #classifier = RandomForestClassifier(random_state=41)
        #classifier = LogisticRegression(random_state=41, max_iter=10000, C=0.2)
        #classifier = GradientBoostingClassifier(random_state=41, learning_rate=0.01)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        #Baseline:
        #y_pred = y_train[-51:]
        score = f1_score(y_test, y_pred)
        time_series_f1_sum += score
        print("Time series score:", score)
        n += 1

    print("Time series: Average f1 score acress all time periods:", time_series_f1_sum / n)

# Inclusive range
def date_iter(start_date, end_date):
    date = start_date
    while date <= end_date:
        yield date
        date += timedelta(days=1)

def run_words_classifier(training_periods, tweets_by_day_by_state, tokenizer):
    f1_sum = 0
    n = 0
    k = 2
    for j in range(k, len(training_periods)):
        prev_periods = training_periods[j-k:j]
        curr_period = training_periods[j]
        docs_train = []
        y_train = []
        for curr_start_date, curr_end_date in prev_periods:
            for date in date_iter(curr_start_date, curr_end_date):
                tweets_by_state = tweets_by_day_by_state.get(date)
                if not tweets_by_state:
                    print("No tweets for:", date)
                    continue
                for state, tweets in tweets_by_state.items():
                    # Label is mapped to the period start date
                    label = state_classes_by_date[curr_start_date][state]
                    for tweet in tweets:
                        docs_train.append(tweet)
                        y_train.append(label)

        docs_test = []
        y_test = []
        test_start_date, test_end_date = curr_period
        for date in date_iter(test_start_date, test_end_date):
            tweets_by_state = tweets_by_day_by_state.get(date)
            if not tweets_by_state:
                print("No tweets for:", date)
                continue
            for state, tweets in tweets_by_state.items():
                label = state_classes_by_date[test_start_date][state]
                for tweet in tweets:
                    docs_test.append(tweet)
                    y_test.append(label)

        text_vectorizer = CountVectorizer(tokenizer=tokenizer, preprocessor=dummy)
        X_train = text_vectorizer.fit_transform(docs_train)
        X_test = text_vectorizer.transform(docs_test)

        #classifier = LinearSVC(random_state=41, max_iter=10000, C=0.001)
        #classifier = RandomForestClassifier(random_state=41)
        #classifier = LogisticRegression(random_state=41, max_iter=10000, C=0.001)
        classifier = GradientBoostingClassifier(random_state=41, learning_rate=0.1)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        score = f1_score(y_test, y_pred)
        f1_sum += score
        print("Words classifier score:", score)
        n += 1

    print("Words: Average f1 score acress all time periods:", f1_sum / n)

training_periods = survey_period_batch_1
run_time_series_classifier(training_periods)

print("Reading tweets...")
tweets_by_day_by_state = read_tweets_by_day_by_state()
print("Starting predictions...")
run_words_classifier(training_periods, tweets_by_day_by_state, tokenize_on_spaces)

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
