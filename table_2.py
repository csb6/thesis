import parser, traceback
from datetime import datetime

class TimePeriod:
    def __init__(self, year, start_pos):
        self.year = year
        self.start_date = datetime(year, 3, 15).timestamp()
        self.end_date = datetime(year, 5, 15).timestamp()
        self.start_pos = start_pos

    def get(self):
        return (self.year, self.start_date, self.end_date, self.start_pos)

timespans = [TimePeriod(2015, 2014015114), \
             TimePeriod(2016, 3573927206), \
             TimePeriod(2017, 4829783454), \
             TimePeriod(2018, 6244851169), \
             TimePeriod(2019, 7196276415), \
             TimePeriod(2020, 7848556306), \
             TimePeriod(2021, 8368962115)]

def main():
    post_count = 0
    timespan_iter = iter(timespans)
    with open("food_sample_2Oct2013_1Sep2021.txt") as big_file:
        year, start_time, end_time, start_pos = next(timespan_iter).get()
        print("Processing:", year)
        big_file.seek(start_pos)
        try:
            for user_metadata, tweet_metadata, tweet in parser.tweet_iter(big_file):
                if tweet_metadata[parser.Post_Time_Col] < start_time:
                    continue
                elif tweet_metadata[parser.Post_Time_Col] <= end_time:
                    post_count += 1
                    #if post_count == 1:
                    #    print("Seek point for first post:", big_file.tell())
                else:
                    # Past end date; end of processing for this year
                    print("Post count:", post_count)
                    print()
                    year, start_time, end_time, start_pos = next(timespan_iter).get()
                    print("Processing:", year)
                    post_count = 0
                    big_file.seek(start_pos)
        except StopIteration:
            print("Done")
        except Exception:
            print("Error at seek pos:", big_file.tell())
            print(traceback.format_exc())

main()
