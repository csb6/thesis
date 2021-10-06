import csv, datetime

# Population data

Date_Col = 0
State_Col = 1
Total_Cases_Col = 2
New_Cases_Col = 5

Is_State_Col = 3
Name_Col = 4
Pop_Col_2019 = 15

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
                     'Wyoming': "WY", \
                     'Puerto Rico': "PR"}

assert len(statename_to_code) == 52

def get_state_populations():
    state_populations = {}
    with open("us_population_2010_2019.csv") as input_file:
        reader = csv.reader(input_file)
        next(reader) # skip column labels
        for row in reader:
            is_state = int(row[Is_State_Col]) > 0
            if not is_state:
                continue
            state_name = row[Name_Col]
            population = int(row[Pop_Col_2019])
            state_populations[statename_to_code[state_name]] = population
    return state_populations

# COVID data

Area_Count = 60

def to_datetime_obj(date_str):
    month, day, year = [int(item) for item in date_str.split("/", 3)]
    return datetime.datetime(year, month, day)

def by_date_and_state(state_data):
    covid_counts_by_date_and_state = {}
    curr_date = None
    curr_date_data = None
    for date, state_code, new_cases in state_data:
        if date != curr_date:
            assert date not in covid_counts_by_date_and_state
            curr_date = date
            covid_counts_by_date_and_state[curr_date] = {}
            curr_date_data = covid_counts_by_date_and_state[curr_date]

        assert state_code not in curr_date_data
        curr_date_data[state_code] = new_cases
    return covid_counts_by_date_and_state

def normalize_non_states(state_populations, counts_by_date_and_state):
    for date, states_to_counts in counts_by_date_and_state.items():
        states_to_counts["NY"] += states_to_counts["NYC"]
        del states_to_counts["NYC"]
        non_states = [key for key in states_to_counts if key not in state_populations]
        for key in non_states:
            del states_to_counts[key]

def normalize_by_population(state_populations, counts_by_date_and_state):
    for date, states_to_counts in counts_by_date_and_state.items():
        for state_code, count in states_to_counts.items():
            quotient = state_populations[state_code] / 100_000
            states_to_counts[state_code] /= quotient
            states_to_counts[state_code] = round(states_to_counts[state_code], 3)

def classify_by_median(counts_by_date_and_state):
    counts_by_date_and_class = {}
    for date, states_to_counts in counts_by_date_and_state.items():
        rows = list(states_to_counts.values())
        rows.sort()
        daily_median = median(rows)
        above_median = set()
        below_median = set()
        for state_code, count in states_to_counts.items():
            if count >= daily_median:
                above_median.add(state_code)
            else:
                below_median.add(state_code)
        counts_by_date_and_class[date] = (below_median, above_median)
    return counts_by_date_and_class

def median(sorted_rows):
    mid = len(sorted_rows) // 2
    if len(sorted_rows) % 2 == 0:
        return (sorted_rows[mid] + sorted_rows[mid-1]) / 2
    else:
        return sorted_rows[mid]

def main():
    state_populations = get_state_populations()
    covid_data = []
    with open("covid-cases-by-state.csv") as csv_file:
        csv_data = csv.reader(csv_file)
        next(csv_data)
        for row in csv_data:
            if not row[Date_Col] or not row[State_Col] or not row[New_Cases_Col]:
                print("Missing data:", row)
                continue
            date = to_datetime_obj(row[Date_Col])
            new_cases = int(row[New_Cases_Col])
            covid_data.append((date, row[State_Col], new_cases))

        covid_data.sort()

    covid_counts_by_date_and_state = by_date_and_state(covid_data)
    normalize_non_states(state_populations, covid_counts_by_date_and_state)
    normalize_by_population(state_populations, covid_counts_by_date_and_state)
    counts_by_date_and_class = classify_by_median(covid_counts_by_date_and_state)

    below_median, above_median = counts_by_date_and_class[datetime.datetime(2021, 9, 27)]
    print("At/above median:", len(above_median))
    print("Below median:", len(below_median))

    print(covid_counts_by_date_and_state[datetime.datetime(2021, 9, 27)])

main()
