import numpy as np
import pandas as pd


def remove_single_actions(df):
  df = df.drop(df[(df['action_type'] == "clickout item") & (df['step'] == 1)].index)
  return df
  
def remove_test_single_actions(df_test, df_gt):
  df_sessions = df_test.groupby('session_id')

  for group_name, df_group in df_sessions:
    session_len = 0

    for action_index, action in df_group.iterrows():
      session_len = session_len + 1
    
    if session_len == 1:
      df_test = df_test.drop(df_test[df_test['session_id'] == action['session_id']].index)
      df_gt = df_gt.drop(df_gt[df_gt['session_id'] == action['session_id']].index)
      #df = df.drop(df[(df['action_type'] == "clickout item") & (df['step'] == 1)].index)
    
  return df_test, df_gt  
  
def remove_nonitem_actions(df):
  df = df.drop(df[(df['action_type'] != 'interaction item image') & (df['action_type'] != 'interaction item deals') & (df['action_type'] != 'clickout item') & (df['action_type'] != 'search for item')].index)
  return df

def reduce_df(df, dim):
  df = df.head(dim)
  return pd.DataFrame(df)

def get_corpus(df):
  session_id = ''
  temp_session = []
  splitted_sessions = []

  for action_index, action in df.iterrows():
    if session_id == '':
      session_id = action['session_id']

    if session_id != action['session_id']:
      splitted_sessions.append(temp_session)
      temp_session = []

    temp_session.append(action['reference'])
    session_id = action['session_id']

  return splitted_sessions

#gets the training set and splits it in subsessions populated by the item of the action
def prepare_input(df_train):
  training_set = []
  category_set = []
  hotels_window_set = []
  
  df_sessions = df_train.groupby('session_id')

  for group_name, df_group in df_sessions:
    sub_sessions = []
    categories = []
    temp_session = []
    hotels_window = []

    for action_index, action in df_group.iterrows():
      if action['action_type'] == 'clickout item':
        sub_sessions.append(temp_session)
        temp_session.append(action)
        categories.append(action['reference'])
        hotels_window.append(action['impressions'].split('|'))
      else:
        temp_session.append(action)
        
    #training_set.concatenate(sub_sessions)
    #category_set.concatenate(categories)
    #hotels_window_set.concatenate(hotels_window)
    training_set = training_set + sub_sessions
    category_set = category_set + categories
    hotels_window_set = hotels_window_set + hotels_window
    
    
  return training_set, category_set, hotels_window_set