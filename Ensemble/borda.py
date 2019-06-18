import pandas as pd
import numpy as np
from collections import Counter
import operator
import math


MERGE_COLS = ["user_id", "session_id", "timestamp", "step"]

def generate_rranks_range(start, end):
    """Generate reciprocal ranks for a given list length."""

    return 1.0 / (np.arange(start, end) + 1)

def read_into_df(file):
    """Read csv file into data frame."""
    df = (
        pd.read_csv(file)
            .set_index(['user_id', 'session_id', 'timestamp', 'step'])
    )

    return df
def score_submissions(subm_csv, gt_csv, objective_function):
    """Score submissions with given objective function."""

    print(f"Reading ground truth data {gt_csv} ...")
    df_gt = read_into_df(gt_csv)

    print(f"Reading submission data {subm_csv} ...")
    df_subm = read_into_df(subm_csv)
    print('Submissions')
    print(df_subm.head(10))
    # create dataframe containing the ground truth to target rows
    cols = ['reference', 'impressions', 'prices']
    df_key = df_gt.loc[:, cols]

    # append key to submission file
    df_subm_with_key = df_key.join(df_subm, how='inner')
    print(df_subm_with_key.head())
    df_subm_with_key.reference = df_subm_with_key.reference.astype(int)
    df_subm_with_key = convert_string_to_list(
        df_subm_with_key, 'item_recommendations', 'item_recommendations'
    )

    # score each row
    df_subm_with_key['score'] = df_subm_with_key.apply(objective_function, axis=1)
    df_subm_with_key.to_csv('borda.csv')
    print(df_subm_with_key)
    mrr = df_subm_with_key.score.mean()

    return mrr

def get_reciprocal_ranks(ps):
    """Calculate reciprocal ranks for recommendations."""
    mask = ps.reference == np.array(ps.item_recommendations)

    if mask.sum() == 1:
        rranks = generate_rranks_range(0, len(ps.item_recommendations))
        return np.array(rranks)[mask].min()
    else:
        return 0.0

def convert_string_to_list(df, col, new_col):
    """Convert column from string to list format."""
    fxn = lambda arr_string: [int(item) for item in str(arr_string).split(" ")]

    mask = ~(df[col].isnull())

    df[new_col] = df[col]
    df.loc[mask, new_col] = df[mask][col].map(fxn)

    return df

def calculate_single_list_score(l):
    """
        Input -> list of string
        Output -> Dictionary {'item': score}
    """
    score_dic = {}
    i = 0
    for rec in l:
        score_dic[rec] = len(l) - i
        i = i + 1
    return score_dic

def sum_and_sort_dictionaries(dic_1, dic_2):
    """
        Input -> 2 dictionaries
        Output -> 1 list of item sorted by score
    """
    sum_dic = dict(Counter(dic_1)+Counter(dic_2))
    sorted_x = sorted(sum_dic.items(), key=operator.itemgetter(1), reverse = True)
    sorted_items = list(map(lambda x:x[0], sorted_x))
    return sorted_items

def calculate_borda(mf_rec, rnn_rec):
    if(mf_rec == ''):
        return rnn_rec
    if(rnn_rec == ''):
        return mf_rec

    # Calculate score dictionary for mf
    mf_rec_dic = calculate_single_list_score(mf_rec.split(' '))
    rnn_rec_dic = calculate_single_list_score(rnn_rec.split(' '))
    list_items = sum_and_sort_dictionaries(mf_rec_dic, rnn_rec_dic)
    result = " ".join(list_items)
    return result

df_mf = pd.read_csv('submission_matrixfactorization_1.csv')
df_rnn = pd.read_csv('submission_rnn_1.csv')
gt_file = 'gt.csv'
submission_file = 'submission_ensemble.csv'


df_merged = (
    df_mf
    .merge(df_rnn,suffixes=('_mf', '_rnn'),
           left_on=MERGE_COLS,
           right_on=MERGE_COLS,
           how="left")
    )
#print(df_merged)
df_merged = df_merged.fillna('')
#print(df_merged)
df_merged['item_recommendations'] = df_merged.apply(lambda x: calculate_borda(x.item_recommendations_mf, x.item_recommendations_rnn), axis=1)
df_merged = df_merged[MERGE_COLS + ['item_recommendations']]
df_merged.to_csv(submission_file)
mrr =score_submissions(submission_file, gt_file, get_reciprocal_ranks)
#print(df_merged.head())
print('Score: ' + str(mrr))