# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import csv
import sys as sys
from numpy import array
from numpy import argmax
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from collections import defaultdict

import ds_manipulation as dsm
import w2vec as w2v
import test_f as tst
import LSTM as lstm
import LSTMParameters as LSTMParam

import argparse

#python3 python3 setup.py --train ./train_1.csv --test ./test_1.csv --gt ./gt_1.csv --epochs 10 --ncomponents 100 --window 3 --learnrate 0.001
torch.manual_seed(1)

'''
STEP 1: IMPORTING and MANIPULATING DATASET
'''

#importing encode set
df_encode = pd.read_csv('./encode_1.csv')
df_encode = dsm.remove_single_actions(df_encode)
df_encode = dsm.remove_nonitem_actions(df_encode)
#df_encode = dsm.reduce_df(df_encode, 10000)


#importing training set
df_train = pd.read_csv('./train_1.csv')
df_train = dsm.remove_single_actions(df_train)
df_train =  dsm.remove_nonitem_actions(df_train)
#df_train = dsm.reduce_df(df_train, 100)

#importing test set
df_test = pd.read_csv('./test_1.csv')
df_test = dsm.remove_single_actions(df_test)
df_test = dsm.remove_nonitem_actions(df_test)
#df_test = dsm.reduce_df(df_test, 100)

#importing ground truth
df_gt = pd.read_csv('./gt_1.csv')
#df_gt = dsm.reduce_df(df_gt, 100)

df_test, df_gt = dsm.remove_test_single_actions(df_test, df_gt)

#importing hotel prices
df_prices = pd.read_csv("./hotel_prices.csv")

price_dict = dsm.get_hotel_prices(df_prices, n_categories = 100)

#Creating a NaN column for item recommendations
df_test['item_recommendations'] = np.nan

test_dim = len(df_test)
temp_session = []
hotels_window = []
i = 0
print_every = 500
step = 0

print(len(df_encode.groupby('user_id')))
print(len(df_encode.groupby('session_id')))

#splitting in sessions while evaluating recommendations for NaN clickouts
'''for action_index, action in df_test.iterrows():    
    if(action['reference'] != 'unknown'):
        if (action['action_type'] == 'clickout item') & math.isnan(float(action['reference'])):
            hotels_window = action['impressions'].split('|')

            if len(temp_session) != 0:
                df_test.loc[action_index, 'item_recommendations'] = tst.list_to_space_string(action['impressions'].split('|'))

            temp_session.append(action)

        else:
            temp_session.append(action)

    if(i < test_dim-1):
        if action['session_id'] != df_test.iloc[[i + 1]]['session_id'].values[0]:
            step = 0
            #print(temp_session)
            #print(hotels_window)
            #print(p.r)
            temp_session = []
            hotels_window = []

    i = i+1  
    step = step + 1


df_sub = tst.get_submission_target(df_test)

#Removing unnecessary columns
df_sub = df_sub[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]

for action_index, action in df_gt.iterrows():
    if action_index not in df_sub.index.values.tolist():
        df_gt = df_gt.drop(action_index)

mask = df_sub["item_recommendations"].notnull()
df_sub = df_sub[mask]

mrr = tst.score_submissions_no_csv(df_sub, df_gt, tst.get_reciprocal_ranks)
print(mrr)'''