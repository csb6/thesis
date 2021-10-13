import parser
from datetime import datetime

class TimePeriod:
    def __init__(self, start_date, end_date, start_pos):
        self.year = start_date.year
        self.start_date = start_date.timestamp()
        self.end_date = end_date.timestamp()
        self.start_pos = start_pos

    def get(self):
        return (self.year, self.start_date, self.end_date, self.start_pos)

# March 21
def spring(year, start_pos):
    return TimePeriod(datetime(year, 3, 21), datetime(year, 6, 20, 23, 59, 59), \
                      start_pos)

# June 21
def summer(year, start_pos):
    return TimePeriod(datetime(year, 6, 21), datetime(year, 9, 22, 23, 59, 59), \
                      start_pos)

# September 23
def fall(year, start_pos):
    return TimePeriod(datetime(year, 9, 23), datetime(year, 12, 21, 23, 59, 59), \
                      start_pos)

# December 22
def winter(year, start_pos):
    return TimePeriod(datetime(year, 12, 22), datetime(year+1, 3, 20, 23, 59, 59), \
                      start_pos)

             # 2015
timespans = [spring(2015, 2028801333), \
             summer(2015, 2405801696), \
             fall(2015, 2825801771), \
             winter(2015, 3195802091), \
             # 2016
             spring(2016, 3579802248), \
             summer(2016, 3948801951), \
             fall(2016, 4234992070), \
             winter(2016, 4509392931), \
             # 2017
             spring(2017, 4829783454), \
             summer(2017, 5110392203), \
             fall(2017, 5390392139), \
             winter(2017, 5650392088), \
             # 2018
             spring(2018, 6244851169), \
             summer(2018, 6502851609), \
             fall(2018, 6757351558), \
             winter(2018, 6977351350), \
             # 2019
             spring(2019, 7196276415), \
             summer(2019, 7376285172), \
             fall(2019, 7536285591), \
             winter(2019, 7690285718), \
             # 2020
             spring(2020, 7848556306), \
             summer(2020, 7988756613), \
             fall(2020, 8128756601), \
             winter(2020, 8259756637), \
             # 2021
             spring(2021, 8368962115), \
             summer(2021, 8478962311)]

post_count = 0

curr_season = "Spring"


def on_post(timespan, user_metadata, tweet_metadata, tweet, text_file):
    global post_count
    post_count += 1
    if post_count == 1:
        print("Seek point for first post:", text_file.tell())

def on_period_end(timespan):
    global curr_season, post_count
    print("Finished processing", curr_season, timespan.year)
    print(" Post count:", post_count)
    post_count = 0

    if curr_season == "Spring":
        curr_season = "Summer"
    elif curr_season == "Summer":
        curr_season = "Fall"
    elif curr_season == "Fall":
        curr_season = "Winter"
    elif curr_season == "Winter":
        curr_season = "Spring"

def main():
    parser.for_each_post_in_timespans(timespans, on_post, on_period_end)

main()
