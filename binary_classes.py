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
