import numpy as np
import pandas as pd
import math
import torch

def reference_to_str(df):
  df['reference'] = df.apply(lambda x: str(x['reference']), axis=1)
  return df

def remove_single_clickout_actions(df):
  print('Initial size: ' + str(df.shape[0]))
  n_action_session = df.groupby('session_id').size().reset_index(name='n_actions')

  df = (df.merge(n_action_session, left_on='session_id', right_on='session_id', how="left"))

  df = df.drop(df[(df["action_type"] == "clickout item") & (df['n_actions'] == 1)].index)
  print('Final size: ' + str(df.shape[0]))
  del df['n_actions']
  return df
  
def remove_nonitem_actions(df):
  df = df.drop(df[(df['action_type'] != 'interaction item image') & (df['action_type'] != 'interaction item deals') & (df['action_type'] != 'clickout item') & (df['action_type'] != 'search for item')].index)
  return df

def reduce_df(df, dim):
  df = df.head(dim)
  return pd.DataFrame(df)

def get_clickout_data(action, clickout_dict, impression_dict):
  clickout_dict[action.session_id] = action.reference
  impression_dict[action.session_id] = action.impressions.split('|')
  return action.reference

def get_list_session_interactions(group, session_dict):
  session_dict[group.session_id.values[0]] = list(group.reference.values)[-200:]

  return " ".join(list(group.reference.values))

def get_training_input(df_train):
  clickout_dict = {}
  impression_dict = {}
  session_dict = {}

  df_train['step_max'] = df_train[df_train['action_type'] == 'clickout item'].groupby(['session_id'])['step'].transform(max)
  df_train['result'] = df_train[df_train['step'] == df_train['step_max']].apply(lambda x: get_clickout_data(x, clickout_dict, impression_dict), axis = 1)
  df_train_corpus = df_train.groupby('session_id').apply(lambda x: get_list_session_interactions(x, session_dict)).reset_index(name = 'hotel_list')
  df_train = df_train.drop(df_train.index[(df_train['step'] == df_train['step_max']) & (df_train["action_type"] == "clickout item")])

  train_corpus = list(session_dict.values())

  return session_dict, clickout_dict, impression_dict, train_corpus

def get_clickout_data_test(action, clickout_dict, impression_dict):
  if math.isnan(float(action.reference)):
    clickout_dict[action.session_id] = action.step
    impression_dict[action.session_id] = action.impressions.split('|')
  return action.reference

def get_test_input(df_test):
  #Creating a NaN column for item recommendations
  df_test['item_recommendations'] = np.nan

  test_step_clickout_dict = {}
  test_impression_dict = {}
  test_sessions_dict = {}

  df_test['step_max'] = df_test[df_test['action_type'] == 'clickout item'].groupby(['session_id'])['step'].transform(max)
  df_test['result'] = df_test[df_test['step'] == df_test['step_max']].apply(lambda x: get_clickout_data_test(x, test_step_clickout_dict, test_impression_dict), axis = 1)
  df_test = df_test.drop(df_test.index[(df_test['step'] == df_test['step_max']) & (df_test["action_type"] == "clickout item")])   
  df_test_corpus = df_test.groupby('session_id').apply(lambda x: get_list_session_interactions(x, test_sessions_dict)).reset_index(name = 'hotel_list')      

  test_corpus = list(test_sessions_dict.values())

  return test_sessions_dict, test_step_clickout_dict, test_impression_dict, test_corpus

def get_batched_sessions(session_dict, category_dict, batchsize):
  batched_sessions = []
  temp_sessions = []
  for session_id in session_dict.keys():
    if session_id in category_dict:
      temp_sessions.append(session_id)
      if len(temp_sessions) == batchsize:
        batched_sessions.append(temp_sessions)
        temp_sessions = []
  
  if temp_sessions != []:
    batched_sessions.append(temp_sessions)

  return batched_sessions