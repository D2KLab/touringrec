import numpy as np
import pandas as pd
import math
import torch

def reference_to_str(df):
  df['reference'] = df.apply(lambda x: str(x['reference']), axis=1)
  return df

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

def remove_single_actions_opt(df):
  df_sessions = df.groupby('session_id')

  for group_name, df_group in df_sessions:
    session_len = 0

    #for action_index, action in df_group.iterrows():
    #  session_len = session_len + 1
    
    session_len = len(df_group)

    if session_len == 1:
      for action_index, action in df_group.iterrows():
        df = df.drop(df[df['session_id'] == action['session_id']].index)
      
      #df = df.drop(df[df['session_id'] == action['session_id']].index)
      #df_gt = df_gt.drop(df_gt[df_gt['session_id'] == action['session_id']].index)
      
      #df = df.drop(df[(df['action_type'] == "clickout item") & (df['step'] == 1)].index)
    
  return df 

def remove_single_clickout_actions(df):
  print('Initial size: ' + str(df.shape[0]))
  n_action_session = df.groupby('session_id').size().reset_index(name='n_actions')
  #print(n_action_session.head())
  df = (df.merge(n_action_session, left_on='session_id', right_on='session_id', how="left"))
  #print(df.head())
  # df['step_max'] = df.groupby(['session_id'])['step'].transform(max)
  # print(df.head())
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

#get corpus for w2vec encoding
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

    # uncomment this and comment line underneat for shirinking multiple sequential occurrences
    # of hotel to 1 
    '''if len(temp_session) != 0:  
      if action['reference'] != temp_session[-1]:
        temp_session.append(action['reference'])
    else:
      temp_session.append(action['reference'])'''

    temp_session.append(action['reference'])
    session_id = action['session_id']

    # uncomment this to include impression list in the corpus
    '''if action['action_type'] == 'clickout item':
      impressions.append(action['impressions'].split('|'))'''

  return splitted_sessions

def extract_unique_meta(df_meta):
    d = []
    h_feat = df_meta['properties']
    for properties in df_meta['properties']:
      temp = properties.split('|')
      for property in temp:
          d.append(property)
    prop_dict = set(d)
    prop_dict = list(prop_dict)
    return prop_dict

def get_meta_dict(df_meta, hotel_list, meta_list):
  #now I map every hotel with corresponding features
  d = {}
  for index, row in df_meta.iterrows():
    key = row['item_id']
    value = row["properties"]
    temp = value.split('|')
    h_features = []
    for property in temp:
      h_features.append(property)
    d[str(key)] = h_features

  for row in hotel_list:
    key = row
    if key not in d:
      d[str(key)] = []
      
  return d

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

    if(len(sub_sessions) != 0):
      training_set.append(sub_sessions[-1])
      category_set.append(categories[-1])
      hotels_window_set.append(hotels_window[-1])

    # Uncomment this for splitting a single session into multiple clickouts
    '''training_set = training_set + sub_sessions
    category_set = category_set + categories
    hotels_window_set = hotels_window_set + hotels_window'''
    
    
  return training_set, category_set, hotels_window_set

# gets the training set and splits it in subsessions populated by the item of the action
def prepare_input_batched(df_train, batch_size):
  training_set = []
  category_set = []
  hotels_window_set = []

  training_set_batched = []
  category_set_batched = []
  hotels_window_set_batched = []

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
        
    # Uncomment this for splitting a single session into multiple clickouts(comment if underneats)
    '''training_set = training_set + sub_sessions
    category_set = category_set + categories
    hotels_window_set = hotels_window_set + hotels_window'''

    if(len(sub_sessions) != 0):
      training_set.append(sub_sessions[-1])
      category_set.append(categories[-1])
      hotels_window_set.append(hotels_window[-1])
  
  temp_session_batched = []
  temp_category_batched = []
  temp_hotel_window_batched = []
  
  for si, session in enumerate(training_set):
    temp_session_batched.append(session)
    temp_category_batched.append(category_set[si])
    temp_hotel_window_batched.append(hotels_window_set[si])
  
    if len(temp_session_batched) == batch_size:
      training_set_batched.append(temp_session_batched)
      category_set_batched.append(temp_category_batched)
      hotels_window_set_batched.append(temp_hotel_window_batched)
      temp_session_batched = []
      temp_category_batched = []
      temp_hotel_window_batched = []

  if len(temp_session_batched) != 0:
    training_set_batched.append(temp_session_batched)
    category_set_batched.append(temp_category_batched)
    hotels_window_set_batched.append(temp_hotel_window_batched)
    
    
  return training_set_batched, category_set_batched, hotels_window_set_batched


def get_clickout_data(action, clickout_dict, impression_dict):
  clickout_dict[action.session_id] = action.reference
  impression_dict[action.session_id] = action.impressions.split('|')
  return action.reference

def get_list_session_interactions(group, session_dict):
  session_dict[group.session_id.values[0]] = list(group.reference.values)
  return " ".join(list(group.reference.values))

def get_training_input(df_train):
  clickout_dict = {}
  impression_dict = {}
  session_dict = {}

  df_train['step_max'] = df_train[df_train['action_type'] == 'clickout item'].groupby(['session_id'])['step'].transform(max)
  df_train['result'] = df_train[df_train['step'] == df_train['step_max']].apply(lambda x: get_clickout_data(x, clickout_dict, impression_dict), axis = 1)
  #df_train = df_train.drop(df_train.index[math.isnan(df_train['step_max'])])
  df_train = df_train.drop(df_train.index[(df_train['step'] == df_train['step_max']) & (df_train["action_type"] == "clickout item")])
  df_train = df_train.groupby('session_id').apply(lambda x: get_list_session_interactions(x, session_dict)).reset_index(name = 'hotel_list')

  return session_dict, clickout_dict, impression_dict

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
  df_test = df_test.groupby('session_id').apply(lambda x: get_list_session_interactions(x, test_sessions_dict)).reset_index(name = 'hotel_list')      

  return test_sessions_dict, test_step_clickout_dict, test_impression_dict

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