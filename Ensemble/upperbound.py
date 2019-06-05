# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
MERGE_COLS = ["user_id", "session_id", "timestamp", "step"]

def best_score(row):
    gt = row.reference
    if(row.item_recommendations_mf != ''):
        mf_rec_list = row.item_recommendations_mf.split(' ')
        mf_pos = mf_rec_list.index(gt) + 1
    else:
        mf_pos = 999
    if(row.item_recommendations_rnn != ''):
        rnn_rec_list = row.item_recommendations_rnn.split(' ')
        rnn_pos = rnn_rec_list.index(gt) + 1
    else:
        rnn_pos = 999

    best_pos = min(mf_pos, rnn_pos)
    score = 1/best_pos

    return pd.Series(index=MERGE_COLS + ['rank_mf', 'rank_rnn', 'best_score'], data=[row.user_id, row.session_id, row.timestamp, row.step, mf_pos, rnn_pos, score])


df_mf = pd.read_csv('submission_matrixfactorization_1.csv')
df_rnn = pd.read_csv('submission_rnn_1.csv')
df_gt = pd.read_csv('gt.csv')
df_mf = df_mf[MERGE_COLS + ['item_recommendations']]
# Clean step 1
#df_mf = df_mf[df_mf['step'] != 1]
#df_mf = df_mf[df_mf['user_id'] != '764BG6TC2QGT']
df_rnn = df_rnn[MERGE_COLS + ['item_recommendations']]
df_gt = df_gt[MERGE_COLS + ['reference']]

df_merged = (
    df_mf
    .merge(df_rnn,suffixes=('_mf', '_rnn'),
           left_on=MERGE_COLS,
           right_on=MERGE_COLS,
           how="left")
    )

df_merged = (
    df_merged
    .merge(df_gt,
           left_on=MERGE_COLS,
           right_on=MERGE_COLS,
           how="left")
    )
#df_merged = df_merged.dropna()
df_merged = df_merged.fillna('')
print(df_merged)
df_merged = df_merged.apply(lambda x : best_score(x), axis=1)
print(df_merged)
print('Score: ' + str(df_merged['best_score'].mean()))
df_merged.to_csv('upperbound.csv')
