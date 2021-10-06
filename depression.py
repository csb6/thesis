import csv, datetime
from binary_classes import classify_by_median

Indicator_Col = 0
Group_Col = 1
State_Col = 2
# Two main phases conducted
Phase_Col = 4
# Time period within the phase
Period_Col = 5
Start_Date_Col = 7
Score_Col = 9

def to_datetime_obj(date_str):
    month, day, year = [int(item) for item in date_str.split("/", 3)]
    return datetime.datetime(year, month, day)

def by_date_and_state(state_data):
    depression_by_date_and_state = {}
    curr_date = None
    curr_date_data = None
    for date, state, score in state_data:
        if date != curr_date:
            assert date not in depression_by_date_and_state
            curr_date = date
            depression_by_date_and_state[curr_date] = {}
            curr_date_data = depression_by_date_and_state[curr_date]

        assert state not in curr_date_data
        curr_date_data[state] = score
    return depression_by_date_and_state

def main():
    depression_data = []
    with open("depression_survey_data.csv") as csv_file:
        csv_data = csv.reader(csv_file)
        next(csv_data)
        for row in csv_data:
            if row[Indicator_Col] != "Symptoms of Depressive Disorder" \
               or row[Group_Col] != "By State":
                continue
            date = to_datetime_obj(row[Start_Date_Col])
            score = float(row[Score_Col])
            depression_data.append((date, row[State_Col], score))
    depression_data.sort()

    depression_by_date_and_state = by_date_and_state(depression_data)
    depression_by_date_and_class = classify_by_median(depression_by_date_and_state)

    below_median, above_median = depression_by_date_and_class[datetime.datetime(2021, 9, 1)]
    print("At/above median:", len(above_median))
    print("Below median:", len(below_median))

    print(depression_by_date_and_state[datetime.datetime(2021, 9, 1)])
    print()
    print("At/above median states/areas:", above_median)

main()
