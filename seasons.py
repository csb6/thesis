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
timespans = [spring(2015, 2028808830), \
             summer(2015, 2415202330), \
             fall(2015, 2834021120), \
             winter(2015, 3213878123), \
             # 2016
             spring(2016, 3580091313), \
             summer(2016, 3956063644), \
             fall(2016, 4235801712), \
             winter(2016, 4546674691), \
             # 2017
             spring(2017, 4850694419), \
             summer(2017, 5119214979), \
             fall(2017, 5413981325), \
             winter(2017, 5662427657), \
             # 2018
             spring(2018, 6269206081), \
             summer(2018, 6520516548), \
             fall(2018, 6771411360), \
             winter(2018, 6994153592), \
             # 2019
             spring(2019, 7210788125), \
             summer(2019, 7382373769), \
             fall(2019, 7548119129), \
             winter(2019, 7693301002), \
             # 2020
             spring(2020, 7857250194), \
             summer(2020, 7997432869), \
             fall(2020, 8133269478), \
             winter(2020, 8271489464), \
             # 2021
             spring(2021, 8376121327), \
             summer(2021, 8481089485)]

def next_season(curr_season):
    if curr_season == "Spring":
        return "Summer"
    elif curr_season == "Summer":
        return "Fall"
    elif curr_season == "Fall":
        return "Winter"
    elif curr_season == "Winter":
        return "Spring"
    else:
        assert False, "Not a valid season name"
