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

def get_term_ranking(filename, col_num, cnvt_fn, tf_threshold=0):
    term_value_list = []
    term_to_rank = {}
    with open(filename) as input_file:
        for line in input_file:
            row = line.strip().split()
            term = row[Term_Col]
            value = cnvt_fn(row[col_num])
            if int(row[Tf_Plain_Col]) < tf_threshold:
                continue
            term_value_list.append((term, value))
    term_value_list.sort(key=lambda a: a[1], reverse=True)
    for rank, (term, _) in enumerate(term_value_list, start=1):
        term_to_rank[term] = rank
    return term_value_list, term_to_rank

def get_terms_by_rank_diff(term_to_rank_1, term_to_rank_2):
    term_rank_diff_list = []
    for term, orig_rank in term_to_rank_1.items():
        if term not in term_to_rank_2:
            continue
        rank_diff = term_to_rank_2[term] - orig_rank
        term_rank_diff_list.append((term, rank_diff, orig_rank))
    term_rank_diff_list.sort(key=lambda a: a[1])
    return term_rank_diff_list

def filter_by_food_words(healthy_foods, neutral_foods, unhealthy_foods, term_rank_diff_list):
    for term, rank_diff, orig_rank in term_rank_diff_list:
        if term in healthy_foods or term in unhealthy_foods or term in neutral_foods:
            yield term, rank_diff, orig_rank

def get_biggest_rank_change_terms(col_num, cnvt_fn, n):
    threshold = 120
    term_value_list_2018, term_to_rank_2018 = get_term_ranking("tf-idf-2018-3-1-thru-2019-9-1.txt", col_num, cnvt_fn, tf_threshold=threshold)
    term_value_list_2020, term_to_rank_2020 = get_term_ranking("tf-idf-2020-3-1-thru-2021-9-1.txt", col_num, cnvt_fn, tf_threshold=threshold)

    term_rank_diff_list = get_terms_by_rank_diff(term_to_rank_2018, term_to_rank_2020)
    last = term_rank_diff_list[-n:]
    last.sort(key=lambda a: a[1], reverse=True)
    return term_rank_diff_list[:n], last

def get_food_words_rank_change_terms(col_num, cnvt_fn):
    threshold = 120
    term_value_list_2018, term_to_rank_2018 = get_term_ranking("tf-idf-2018-3-1-thru-2019-9-1.txt", col_num, cnvt_fn, tf_threshold=threshold)
    term_value_list_2020, term_to_rank_2020 = get_term_ranking("tf-idf-2020-3-1-thru-2021-9-1.txt", col_num, cnvt_fn, tf_threshold=threshold)

    term_rank_diff_list = get_terms_by_rank_diff(term_to_rank_2018, term_to_rank_2020)
    return [(term, rank, orig_rank) for term, rank, orig_rank in term_rank_diff_list \
            if term in healthy_foods \
            or term in neutral_foods \
            or term in unhealthy_foods]

def print_rank_change(results, label1, label2):
    print(label1, len(results), "terms by change in rank by", label2,
          "in 2018-3-1 to 2019-9-1 vs. 2020-3-1 to 2021-9-1")
    for i, (term, rank_diff, orig_rank) in enumerate(results, start=1):
        print(i, term, rank_diff, orig_rank)

def main():
    low_100_tf, top_100_tf = get_biggest_rank_change_terms(Tf_Plain_Col, lambda x: int(x), 100)
    print_rank_change(top_100_tf, "Top", "tf")
    print_rank_change(low_100_tf, "Bottom", "tf")

    low_100_df, top_100_df = get_biggest_rank_change_terms(Df_Col, lambda x: float(x), 100)
    print_rank_change(top_100_df, "Top", "df")
    print_rank_change(low_100_df, "Bottom", "df")

    low_100_tf_log_idf_log, top_100_tf_log_idf_log = \
        get_biggest_rank_change_terms(Tf_Log_Idf_Log_Col, lambda x: float(x), 100)
    print_rank_change(top_100_tf_log_idf_log, "Top", "(1+log(tf))*log(idf)")
    print_rank_change(low_100_tf_log_idf_log, "Bottom", "(1+log(tf))*log(idf)")

    #tf_ranking = get_food_words_rank_change_terms(Tf_Plain_Col, lambda x: int(x))
    #print_rank_change(tf_ranking, "Food words", "tf")

    #df_ranking = get_food_words_rank_change_terms(Df_Col, lambda x: float(x))
    #print_rank_change(df_ranking, "Food words", "df")

    #tf_log_idf_log_ranking = \
    #    get_food_words_rank_change_terms(Tf_Log_Idf_Log_Col, lambda x: float(x))
    #print_rank_change(tf_log_idf_log_ranking, "Food words", "(1+log(tf))*log(idf)")

main()
