import pandas as pd
import numpy as np
import functions as f
import argparse

def remove_null_clickout(df):
    """
    Remove all the occurences where the clickout reference is set to null (Item to predict)
    """
    df = df.drop(df[(df['action_type'] == "clickout item") & (df['reference'].isnull())].index)
    return df

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

df_train = pd.read_csv('train_' + percentage_s + '.csv')

#creating encode set by removing null clickouts from test set
df_test_no_null = remove_null_clickout(df_cleaned)
df_encode = pd.concat([df_train, df_test_no_null])
df_encode.to_csv('encode_' + percentage_s + '.csv')