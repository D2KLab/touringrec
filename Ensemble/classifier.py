import pandas as pd
import numpy as np
import sys

def get_session_stats(df_gt):
    '''
        Input -> ground truth dataframe
        Output -> df: session_id|session_length|n_hotel|sparsity
    '''
    mask = (df_gt["action_type"] == "clickout item") | (df_gt["action_type"] == "interaction item rating") | (df_gt["action_type"] == "search for item")|(df_gt["action_type"] == "interaction item image") | (df_gt["action_type"] == "interaction item deals")
    df_gt = df_gt[mask] 
    s = df_gt.groupby('session_id').agg({
        'session_id': "count",
        'reference': 'nunique'
    })
    s.columns = ['session_length', 'n_hotel']
    s['sparsity'] = s.apply(lambda x: x.n_hotel / x.session_length, axis=1)
    return s

def calculate_best(rank_mf, rank_rnn):
    if(rank_mf > rank_rnn):
        return 0
    else:
        return 1

def generate_labels(df):
    df['label'] = df.apply(lambda x: calculate_best(x.rank_mf, x.rank_rnn), axis = 1)
    return df


df_gt = pd.read_csv('gt.csv')
df_label = pd.read_csv('upperbound.csv')
session_stats = get_session_stats(df_gt)
df_merged = (
    df_label
    .merge(session_stats,suffixes=('_mf', '_rnn'),
           left_on='session_id',
           right_on='session_id',
           how="left")
    )
print('Remove ties')
df_merged = df_merged[df_merged['rank_mf'] != df_merged['rank_rnn']]
print('Remove single click')
df_merged = df_merged[df_merged['session_length'] != 1]
df_input_svm = generate_labels(df_merged)
df_input_svm = df_input_svm[['session_id', 'session_length', 'sparsity', 'label']]
print(df_input_svm.head())