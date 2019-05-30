import pandas as pd
import numpy as np

def get_interaction_actions(df, actions =["clickout item", "interaction item rating", "search for item", "interaction item image", "interaction item deals", "interaction item info"],clean_null=False):
    """ Return a dataset where all the actions are related to interaction between user and item.
    actions -> list of user/item interactions to select
    clean_null -> if is set to true cleans the Null clickout actions of the test set"""
    
    #mask = (df["action_type"] == "clickout item") | (df["action_type"] == "interaction item rating") | (df["action_type"] == "search for item")|(df["action_type"] == "interaction item image") | (df["action_type"] == "interaction item deals")
    
    df_cleaned = df[df['action_type'].isin(actions)]

    if 'interaction item info' in actions:
        df_cleaned = df_cleaned.drop(df_cleaned[(df_cleaned['action_type'] == 'interaction item info') & (df_cleaned['reference'].str.isdigit() != True)].index)
    
    if clean_null:
        df_cleaned = df_cleaned.drop(df_cleaned[(df_cleaned['action_type'] == "clickout item") & (df_cleaned['reference'].isnull())].index)
    
    return df_cleaned

def remove_null_clickout(df):
    """
    Remove all the occurences where the clickout reference is set to null (Item to predict)
    """
    df = df.drop(df[(df['action_type'] == "clickout item") & (df['reference'].isnull())].index)
    return df

def split_one_action(df_test):
    """
    Required Input -
        - df_test = test set dataframe
    Expected Output  -
        - df_no_single_action = dataframe without clickout at step 1 to predict
        - df_single_action = dataframe with only clickout at step 1 to predict
    """
    df_test = f.get_submission_target(df_test)
    df_no_single_action = remove_single_actions(df_test)
    df_single_action = get_single_actions(df_test)
    print('Total item of test set: ' + str(df_test.shape[0]) + ' No single action: #' + str(df_no_single_action.shape[0]) + ' Only single actions: #' + str(df_single_action.shape[0]))
    return df_no_single_action, df_single_action

def remove_single_actions(df):
    df = df.drop(df[(df['action_type'] == "clickout item") & (df['step'] == 1) & (df['reference'].isnull())].index)
    return df

def get_single_actions(df):
    df = df[(df['action_type'] == "clickout item") & (df['step'] == 1) & (df['reference'].isnull())]
    return df

def clean_no_clickout_session(df):
    '''
    Required input:
        -df -> pandas dataframe
    Expected output:
        -df -> same dataframe, without all the session that does not have a clickout
    '''
    df_clickout = df[df['action_type'] == 'clickout item']
    print('Before: ' + str(df.shape[0]))
    df_cleaned = df[df['session_id'].isin(df_clickout['session_id'].drop_duplicates())]
    print('After: ' + str(df_cleaned.shape[0]))
    return df_cleaned