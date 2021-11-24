import parser

# Column numbers
Term_Col = 0
Tf_Plain_Col = 1
Tf_Log_Col = 2
Df_Col = 3
Idf_Plain = 4
Idf_Log = 5
Tf_Log_Idf_Log_Col = 9

healthy_foods, neutral_foods, unhealthy_foods = [set(x) for x in parser.get_food_scores("food_scores.txt")]

def get_term_ranking(filename, col_num, cnvt_fn, tf_threshold=5):
    term_value_list = []
    term_to_rank = {}
    with open(filename) as input_file:
        for line in input_file:
            row = line.strip().split()
            term = row[Term_Col]
            value = cnvt_fn(row[col_num])
            if int(row[Tf_Plain_Col]) < tf_threshold:
                continue
            elif term not in healthy_foods and term not in neutral_foods \
                 and term not in unhealthy_foods:
                continue
            term_value_list.append((term, value))
    # Sort in descending order by the chosen statistic's value
    term_value_list.sort(key=lambda term_value: term_value[1], reverse=True)
    # Rank (from highest value to lowest value for the statistic) each term
    for rank, (term, _) in enumerate(term_value_list, start=1):
        term_to_rank[term] = rank
    return term_value_list, term_to_rank

def by_rank_diff(data):
    term, orig_rank, new_rank = data
    return new_rank - orig_rank

def get_terms_by_rank_diff(term_to_rank_1, term_to_rank_2):
    term_rank_diff_list = []
    for term, orig_rank in term_to_rank_1.items():
        if term not in term_to_rank_2:
            continue
        new_rank = term_to_rank_2[term]
        term_rank_diff_list.append((term, orig_rank, new_rank))
    term_rank_diff_list.sort(key=by_rank_diff)
    return term_rank_diff_list

def print_ranks(results, label):
    print("Terms by change in rank by", label,
          "in 2018-3-1 to 2019-9-1 vs. 2020-3-1 to 2021-9-1")
    for term, orig_rank, new_rank in results:
        if term in healthy_foods:
            kind = "healthy"
        elif term in neutral_foods:
            kind = "neutral"
        else:
            kind = "unhealthy"
        print(term, kind, orig_rank, new_rank)

def get_stats(terms_by_rank_diff):
    healthy_food_sum = 0
    healthy_food_count = 0
    neutral_food_sum = 0
    neutral_food_count = 0
    unhealthy_food_sum = 0
    unhealthy_food_count = 0
    for term, orig_rank, new_rank in terms_by_rank_diff:
        diff = new_rank - orig_rank # orig_rank + diff = new_rank
        if term in healthy_foods:
            healthy_food_sum += diff
            healthy_food_count += 1
        elif term in neutral_foods:
            neutral_food_sum += diff
            neutral_food_count += 1
        else:
            unhealthy_food_sum += diff
            unhealthy_food_count += 1
    return healthy_food_sum / healthy_food_count, \
        neutral_food_sum / neutral_food_count, \
        unhealthy_food_sum / unhealthy_food_count

def main():
    term_value_list_2018, term_to_rank_2018 = \
        get_term_ranking("tf-idf-2018-3-1-thru-2019-9-1.txt", Tf_Plain_Col, \
                         lambda x: int(x))
    term_value_list_2020, term_to_rank_2020 = \
        get_term_ranking("tf-idf-2020-3-1-thru-2021-9-1.txt", Tf_Plain_Col, \
                         lambda x: int(x))

    terms_by_rank_diff = get_terms_by_rank_diff(term_to_rank_2018, term_to_rank_2020)

    print_ranks(terms_by_rank_diff, "tf")

    healthy_avg_diff, neutral_avg_diff, unhealthy_avg_diff = get_stats(terms_by_rank_diff)
    print("Average rank difference for healthy food words:", healthy_avg_diff)
    print("Average rank difference for neutral food words:", neutral_avg_diff)
    print("Average rank difference for unhealthy food words:", unhealthy_avg_diff)

    """low_100_df, top_100_df = get_biggest_rank_change_terms(Df_Col, lambda x: float(x), 100)
    print_rank_change(top_100_df, "Top", "df")
    print_rank_change(low_100_df, "Bottom", "df")

    low_100_tf_log_idf_log, top_100_tf_log_idf_log = \
        get_biggest_rank_change_terms(Tf_Log_Idf_Log_Col, lambda x: float(x), 100)
    print_rank_change(top_100_tf_log_idf_log, "Top", "(1+log(tf))*log(idf)")
    print_rank_change(low_100_tf_log_idf_log, "Bottom", "(1+log(tf))*log(idf)")"""

main()
