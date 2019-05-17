import pandas as pd
import numpy as np
import functions as f
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--percentage', action="store", type=int, help="Define the percentage of the dataset that you want to split")
args = parser.parse_args()
percentage = args.percentage
if percentage == None:
    percentage_s=""
    percentage = 1
else:
    percentage_s = str(percentage)
    percentage = percentage / 100

print('Reading the training set')
df_train = pd.read_csv('train_off.csv')
df_train = f.get_df_percentage(df_train, percentage)

session_train, session_test = f.split_group_random(df_train, 0.8, 'session_id')

f.split_group_csv(df_train, session_train, session_test, 'session_id', percentage=percentage_s)

df_test = pd.read_csv('gt_' + percentage_s + '.csv')
df_cleaned = f.clean_dataset(df_test)
df_cleaned.to_csv('test_' + percentage_s + '.csv', index= False)