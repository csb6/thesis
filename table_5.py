import parser, traceback
from datetime import datetime, date

class TimePeriod:
    def __init__(self, year, start_pos):
        self.year = year
        self.start_date = datetime(year, 3, 15).timestamp()
        self.end_date = datetime(year, 5, 15).timestamp()
        self.start_pos = start_pos

    def get(self):
        return (self.year, self.start_date, self.end_date, self.start_pos)

timespans = [TimePeriod(2014, 300000185), \
             TimePeriod(2015, 2014015114), \
             TimePeriod(2016, 3573927206), \
             TimePeriod(2017, 4829783454), \
             TimePeriod(2018, 6244851169), \
             TimePeriod(2019, 7196276415), \
             TimePeriod(2020, 7848556306), \
             TimePeriod(2021, 8368962115)]

northeast = {"ME", "NH", "CT", "NJ", "PA", "NY", "VT", "RI", "MA"}
midwest = {"ND", "SD", "NE", "KS", "MN", "IA", "MO", "WI", "IL", "IN", "MI", "OH"}
south = {"OK", "TX", "AR", "LA", "TN", "MS", "AL", "GA", "FL", "SC", "NC", \
         "VA", "WV", "MD", "DC", "DE", "KY", "TN"}
west = {"WA", "OR", "CA", "NV", "AZ", "UT", "CO", "NM", "ID", "MT", "WY", \
        "HI", "AK"}

assert len(northeast) + len(midwest) + len(south) + len(west) == 51

healthy_foods, neutral_foods, unhealthy_foods = parser.get_food_scores("food_scores.txt")

def main():
    timespan_iter = iter(timespans)
    with open("food_sample_2Oct2013_1Sep2021.txt") as big_file:
        year, start_time, end_time, start_pos = next(timespan_iter).get()
        unhealthy_food_counts = {"northeast": 0, "midwest": 0, "south": 0, "west": 0}
        total_food_counts = {"northeast": 0, "midwest": 0, "south": 0, "west": 0}
        print("Processing:", year)
        big_file.seek(start_pos)
        found_first = False
        try:
            for user_metadata, tweet_metadata, tweet in parser.tweet_iter(big_file):
                if tweet_metadata[parser.Post_Time_Col] < start_time:
                    continue
                elif tweet_metadata[parser.Post_Time_Col] <= end_time:
                    if not found_first:
                        print("Seek pos for start of", year, "is", big_file.tell())
                        found_first = True
                    tokens = parser.tokenize(tweet)
                    state_code = user_metadata[parser.User_Location_Col]
                    if not state_code:
                        continue
                    elif state_code in northeast:
                        region = "northeast"
                    elif state_code in midwest:
                        region = "midwest"
                    elif state_code in south:
                        region = "south"
                    elif state_code in west:
                        region = "west"
                    else:
                        print("Error: unknown state/area:", state_code)
                        continue
                    for token in tokens:
                        if token in unhealthy_foods:
                            unhealthy_food_counts[region] += 1
                            total_food_counts[region] += 1
                        elif token in healthy_foods:
                            total_food_counts[region] += 1
                else:
                    # Past end date; end of processing for this year
                    print("Unhealthy food words as percentage of all food words:")
                    for region, unhealthy_count in unhealthy_food_counts.items():
                        print(" ", region, unhealthy_count / total_food_counts[region])

                    year, start_time, end_time, start_pos = next(timespan_iter).get()
                    unhealthy_food_counts = {"northeast": 0, "midwest": 0, "south": 0, "west": 0}
                    total_food_counts = {"northeast": 0, "midwest": 0, "south": 0, "west": 0}
                    print("Processing:", year)
                    big_file.seek(start_pos)
        except StopIteration:
            print("Done.")

main()
