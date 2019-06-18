import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import csv
from tqdm import tqdm
from pathlib import Path



"""
Input: pandas Dataframe, percentage of training data, parameter to split
Output: two list of session_id (train and test)
"""
def split_group_random(df, perc, split_param):
    print('Loading the dataset...')
    array_sID = df[[split_param]].values
    array_sID = np.unique(array_sID)
    print('Generate the two splitted lists...')
    if isinstance(perc, float):
        if(perc >= 0 and perc <=1):
            x_train, x_test = train_test_split(array_sID, train_size=perc)
        else:
            raise Exception('The value of split must be between 0 and 1, current value: ' + perc)
    else:
        raise Exception('The percentage must be a float, current type: ' + type(perc))

    return set(x_train), set(x_test)

"""
Input: path to csv, list of the data that should be on the training set, column name of the group to split
Output: two different csv for train and test
TODO: parametrizzare (csv as dictionary?)
"""
"""
def split_group_csv(path, list_train, list_test, split_param):
    train_file = open('train.csv', mode='w')
    test_file = open('test.csv', mode='w')
    csv_file = open(path, mode='r')
    csv_reader = csv.reader(csv_file)
    train_writer = csv.writer(train_file, delimiter=',')
    test_writer = csv.writer(test_file, delimiter=',')
        
    next(csv_reader) # Skip the first line that is an header
    for row in csv_reader:
        if str(row[1]) in list_train: #write in the training set
            train_writer.writerow(row)
        elif str(row[1]) in list_test: #write in the test set
            test_writer.writerow(row)
        else:
            raise Exception('The session is not included in one of the 2 list!')
    
    train_file.close()
    test_file.close()
    csv_file.close()
    return
"""


def split_group_csv(df, list_train, list_test, split_param, percentage=""):

    if Path("test_" + percentage + '.csv').is_file():
        os.remove("test_" + percentage + '.csv')
        print('Cleared test_' + percentage + '.csv')
    if Path("gt_" + percentage + '.csv').is_file():
        os.remove("gt_" + percentage + '.csv')
        print('Cleared gt_' + percentage + '.csv')
    if Path("train_" + percentage + '.csv').is_file():
        os.remove("train_" + percentage + '.csv')
        print('Cleared train_' + percentage + '.csv')
    
    print('Start writing the file...')

    train_file = open('train_' + percentage + '.csv', mode='w')
    train_writer = csv.writer(train_file, delimiter=',')
    test_file = open('gt_' + percentage + '.csv' , mode='w')
    test_writer = csv.writer(test_file, delimiter=',')
    #Write headers
    train_writer.writerow(df.columns.values)
    test_writer.writerow(df.columns.values)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row[split_param] in list_train:
            # print('Row #' + str(index) + ' of: ' + str(df.shape[0]) + ' on TRAIN')
            #row.to_csv(train_file, mode='a', header=True, index=False)
            train_writer.writerow(row.values)
        elif row[split_param] in list_test:
            # print('Row #' + str(index) + ' of: ' + str(df.shape[0]) + ' on TEST')
            #row.to_csv(test_file, mode='a', header=True, index=False)
            test_writer.writerow(row.values)
        else:
            raise Exception('The session is not included in one of the 2 list!')
    train_file.close()
    test_file.close()
    return

def clean_dataset(df_test):
    df_test['step_max'] = df_test.groupby(['user_id'])['step'].transform(max)
    df_test.loc[(df_test['step_max'] == df_test['step']) & (df_test["action_type"] == "clickout item"), ["reference"]] = None
    del df_test['step_max']
    return df_test

def get_df_percentage(df, perc):
    df_size = df.shape[0]
    return df.head(int(perc * df_size))