import numpy as np
import pandas as pd
import torch
import math

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
  impressions = []

  for action_index, action in df.iterrows():
    if session_id == '':
      session_id = action['session_id']

    if session_id != action['session_id']:
      splitted_sessions.append(temp_session)
      #splitted_sessions = splitted_sessions + impressions
      temp_session = []
      #impressions = []

    temp_session.append(action['reference'])
    session_id = action['session_id']
    
    #if action['action_type'] == 'clickout item':
      #impressions.append(action['impressions'].split('|')[:8])

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
        
    #training_set.concatenate(sub_sessions)
    #category_set.concatenate(categories)
    #hotels_window_set.concatenate(hotels_window)
    training_set = training_set + sub_sessions
    category_set = category_set + categories
    hotels_window_set = hotels_window_set + hotels_window
    
    
  return training_set, category_set, hotels_window_set

  #gets the training set and splits it in subsessions populated by the item of the action
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
        
    #training_set.concatenate(sub_sessions)
    #category_set.concatenate(categories)
    #hotels_window_set.concatenate(hotels_window)
    training_set = training_set + sub_sessions
    category_set = category_set + categories
    hotels_window_set = hotels_window_set + hotels_window
  
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

  def generate_prices_sparse_matrix(df, features_col='intervals'):
    df['present'] = 1
    hotel_dict = create_item_dict(df) #Controllare che sia uguale all'altro dizionario
    feature_dict = create_item_dict(df, col_name='feature')
    list_hotel = list(df['reference'])
    list_features = list(df['feature'])
    list_data = list(df['present'])
    n_items = len(list_hotel)
    n_features = len(list_features)
    # Convert each list of string in a list of indexes
    list_items = list(map(lambda x: hotel_dict[x], list_hotel))
    list_features = list(map(lambda x: feature_dict[x], list_features))
    # Generate the sparse matrix
    row = np.array(list_items)
    col = np.array(list_features)
    data = np.array(list_data)
    csr = csr_matrix((data, (row, col)), shape=(n_items, n_features))

    return csr, hotel_dict
  
def get_hotel_prices(df_metadata, n_categories = 2000):
    """
    Required Input -
        - metadata_file = file with the average price for each hotel
    """
    #print("Reading metadata: " + metadata_file)
    df_metadata['price'] = df_metadata['price'].apply(lambda x: math.log10(x))
    # Define the range
    max_price = df_metadata['price'].max()
    min_price = df_metadata['price'].min()
    range = (max_price - min_price) / n_categories
    # Generate the classes
    df_metadata['intervals'] = pd.cut(df_metadata['price'], bins=np.arange(min_price,max_price,range))
    df_metadata.loc[:, 'intervals'] = df_metadata['intervals'].apply(str)
    #classes_dic = create_user_dict(df_metadata, col_name = 'intervals')
    #df_metadata.loc[:, 'intervals'] = df_metadata['intervals'].apply(lambda x : classes_dic.get(x))
    #df_metadata.loc[:, 'intervals'] = df_metadata['intervals'].apply(int)
    # Create a dictionary of item_id -> price_category
    price_dic = pd.Series(df_metadata.intervals.values,index=df_metadata.impressions).to_dict()

    return price_dic