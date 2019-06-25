import csv
import torch
import numpy as np
import pandas as pd
import functions as f
import LSTM as lstm
import math as math
from operator import itemgetter


def list_to_space_string(l):
    """Return a space separated string from a list"""
    s = " ".join(l)
    return s

def recommendations_from_output(output, hotel_dict, hotels_window, n_features, prev_hotel):
  i = 0
  window_dict = {}
  
  prev_hotel = prev_hotel[0]
  
  output_arr = np.asarray(output[0].cpu().detach().numpy())

  sub_hotels = []
  
  out_class_v, out_class_i = torch.max(output, 1)
  
  if out_class_i == 0:
    sub_hotels = hotels_window
  else:
    sub_hotels.append(prev_hotel)
    for hotel in hotels_window:
      if hotel != sub_hotels[0]:
        sub_hotels.append(hotel) 
                           
  return list_to_space_string(sub_hotels)

def evaluate(model, session, hotel_dict, n_features, hotels_window, max_window, meta_dict, meta_list, prev_hotel):
    """Just return an output list of hotel given a single session."""
    
    session_tensor = lstm.session_to_tensor(session, hotel_dict, n_features, hotels_window, max_window, meta_dict, meta_list)
    
    output = model(session_tensor)

    output = recommendations_from_output(output, hotel_dict, hotels_window, n_features, prev_hotel)

    return output
  
def get_submission_target(df):
    """Identify target rows with missing clickouts."""

    mask = df["reference"].isnull() & (df["action_type"] == "clickout item")
    df_out = df[mask]

    return df_out  

def get_reciprocal_ranks(ps):
    """Calculate reciprocal ranks for recommendations."""
    mask = ps.reference == np.array(ps.item_recommendations)

    if mask.sum() == 1:
        rranks = generate_rranks_range(0, len(ps.item_recommendations))
        return np.array(rranks)[mask].min()
    else:
        return 0.0
  
def generate_rranks_range(start, end):
    """Generate reciprocal ranks for a given list length."""

    return 1.0 / (np.arange(start, end) + 1)
  
def convert_string_to_list(df, col, new_col):
    """Convert column from string to list format."""
    fxn = lambda arr_string: [int(item) for item in str(arr_string).split(" ")]

    mask = ~(df[col].isnull())

    df[new_col] = df[col]
    df.loc[mask, new_col] = df[mask][col].map(fxn)

    return df
  
def read_into_df(file):
    """Read csv file into data frame."""
    df = (
        pd.read_csv(file)
            .set_index(['user_id', 'session_id', 'timestamp', 'step'])
    )

    return df

def score_submissions_no_csv(df_subm, df_gt, objective_function):
    """Return score calculated on given submission dataframe"""
    # create dataframe containing the ground truth to target rows
    cols = ['reference', 'impressions', 'prices']
    df_key = df_gt.loc[:, cols]

    # append key to submission file
    df_subm_with_key = df_key.join(df_subm, how='inner')
    df_subm_with_key.reference = df_subm_with_key.reference.astype(int)
    df_subm_with_key = convert_string_to_list(
        df_subm_with_key, 'item_recommendations', 'item_recommendations'
    )

    #print(df_subm_with_key)

    # score each row
    df_subm_with_key['score'] = df_subm_with_key.apply(objective_function, axis=1)
    mrr = df_subm_with_key.score.mean()

    return mrr
  
def score_submissions(subm_csv, gt_csv, objective_function):
    """Score submissions with given objective function."""

    #print(f"Reading ground truth data {gt_csv} ...")
    df_gt = read_into_df(gt_csv)

    #print(f"Reading submission data {subm_csv} ...")
    df_subm = read_into_df(subm_csv)
    #print('Submissions')
    #print(df_subm.head(10))

    # create dataframe containing the ground truth to target rows
    cols = ['reference', 'impressions', 'prices']
    df_key = df_gt.loc[:, cols]

    # append key to submission file
    df_subm_with_key = df_key.join(df_subm, how='inner')
    df_subm_with_key.reference = df_subm_with_key.reference.astype(int)
    df_subm_with_key = convert_string_to_list(
        df_subm_with_key, 'item_recommendations', 'item_recommendations'
    )

    # score each row
    df_subm_with_key['score'] = df_subm_with_key.apply(objective_function, axis=1)
    mrr = df_subm_with_key.score.mean()

    return mrr

def test_accuracy(model, df_test, df_gt, hotel_dict, n_features, max_window, meta_dict, meta_list, subname="submission_default_name", isprint=False):
    """Return the score obtained by the net on the test dataframe"""

    #Creating a NaN column for item recommendations
    df_test['item_recommendations'] = np.nan

    test_dim = len(df_test)
    temp_session = []
    hotels_window = []
    i = 0
    print_every = 500
    step = 0

    #splitting in sessions while evaluating recommendations for NaN clickouts
    for action_index, action in df_test.iterrows():    
        if(action['reference'] != 'unknown'):
            if (action['action_type'] == 'clickout item') & math.isnan(float(action['reference'])):
                hotels_window = action['impressions'].split('|')

                if len(temp_session) != 0:
                    df_test.loc[action_index, 'item_recommendations'] = evaluate(model, temp_session, hotel_dict, n_features, hotels_window, max_window, meta_dict, meta_list)

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


    df_sub = get_submission_target(df_test)

    #Removing unnecessary columns
    df_sub = df_sub[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]

    for action_index, action in df_gt.iterrows():
        if action_index not in df_sub.index.values.tolist():
            df_gt = df_gt.drop(action_index)

    mask = df_sub["item_recommendations"].notnull()
    df_sub = df_sub[mask]

    # Saving df_sub
    if isprint:
        df_sub.to_csv('./' + subname + '.csv')

    mrr = score_submissions_no_csv(df_sub, df_gt, get_reciprocal_ranks)
    return mrr

def prepare_test(df_test):
  #Creating a NaN column for item recommendations
  df_test['item_recommendations'] = np.nan

  test_dim = len(df_test)

  temp_session = []
  test_sessions = []

  temp_clickout_index = []
  test_clickout_index = []

  hotels_window = []
  test_hotels_window = []

  temp_prev_hotel_list = []
  prev_hotel_list = []

  i = 0
  step = 0

  #splitting in sessions while evaluating recommendations for NaN clickouts
  for action_index, action in df_test.iterrows():
      if(action['reference'] != 'unknown'):
          if (action['action_type'] == 'clickout item') & math.isnan(float(action['reference'])):
            if prev_hotel != '':
              hotels_window = action['impressions'].split('|')
              temp_session.append(action)
              temp_clickout_index.append(action_index)
              temp_prev_hotel_list.append(prev_hotel)
            else:
              temp_session.append(action)
          else:
              temp_session.append(action)
          prev_hotel = action['reference']

      if(i < test_dim-1):
          if action['session_id'] != df_test.iloc[[i + 1]]['session_id'].values[0]:
              step = 0
              test_sessions.append(temp_session)
              test_hotels_window.append(hotels_window)
              test_clickout_index.append(temp_clickout_index)
              prev_hotel_list.append(temp_prev_hotel_list)

              temp_session = []
              hotels_window = []
              temp_clickout_index = []
              temp_prev_hotel_list = []
              prev_hotel = ''


      i = i+1  
      step = step + 1
        
  return test_sessions, test_hotels_window, test_clickout_index, prev_hotel_list
  
  
def test_accuracy_optimized(model, df_test, df_gt, sessions, hotels_window, clickout_index, hotel_dict, n_features, max_window, meta_dict, meta_list, prev_hotel_list, subname="submission_default_name", isprint=False):
  """Return the score obtained by the net on the test dataframe"""

  test_dim = len(df_test)

  print_every = 500


  for session_index, session in enumerate(sessions):
    if clickout_index[session_index] != []:
      df_test.loc[clickout_index[session_index], 'item_recommendations'] = evaluate(model, session, hotel_dict, n_features, hotels_window[session_index], max_window, meta_dict, meta_list, prev_hotel_list[session_index])

  df_sub = get_submission_target(df_test)

  #Removing unnecessary columns
  df_sub = df_sub[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]

  mask = df_sub["item_recommendations"].notnull()
  df_sub = df_sub[mask]

  # Saving df_sub
  if isprint:
      df_sub.to_csv('./' + subname + '.csv')

  mrr = score_submissions_no_csv(df_sub, df_gt, get_reciprocal_ranks)
  return mrr


### FUNCTIONS FOR CLASSIFICATION TASK ###

def recommendations_from_output_classification(output, hotel_dict, hotels_window, n_features, prev_hotel):
  i = 0
  window_dict = {}
  
  prev_hotel = prev_hotel[0]
  
  output_arr = np.asarray(output[0].cpu().detach().numpy())

  sub_hotels = []
  sub_scores = []
  
  out_class_v, out_class_i = torch.max(output, 1)
  if out_class_i == 0:
    for hotel in hotels_window:
      sub_hotels.append(hotel)
      sub_scores.append(0)
  else:
    sub_hotels.append(prev_hotel)
    #print(out_class_v)
    #print(float(out_class_v))
    sub_scores.append(float(out_class_v))

    for hotel in hotels_window:
      if hotel != sub_hotels[0]:
        sub_hotels.append(hotel)
        sub_scores.append(0)

  #print(sub_hotels)
                             
  return sub_hotels, sub_scores
  

# Just return an output given a line
def evaluate_classification(model, session, hotel_dict, n_features, hotels_window, max_window, meta_dict, meta_list, prev_hotel):
    line_tensor = lstm.session_to_tensor(session, hotel_dict, n_features, hotels_window, max_window, meta_dict, meta_list)
    
    output = model(line_tensor)
        
    output, out_scores = recommendations_from_output_classification(output, hotel_dict, hotels_window, n_features, prev_hotel)

    return output, out_scores
  
def test_accuracy_optimized_classification(model, df_test, sessions, hotels_window, clickout_index, hotel_dict, n_features, max_window, meta_dict, meta_list, prev_hotel_list, df_gt = [], subname="submission_default_name", isprint=False, dev = False):
  """Return the score obtained by the net on the test dataframe"""

  test_dim = len(df_test)

  print_every = 500
  
  #missed_target = 0
  if dev:
    fname = 'rnn_test_sub_xgb_2class_dev_10%.csv'
  else:
    fname = 'rnn_test_sub_xgb_2class_inner_10%.csv'  

  with open(fname, mode='w') as test_xgb_sub:
    
    file_writer = csv.writer(test_xgb_sub)
    file_writer.writerow(['session_id', 'hotel_id', 'score'])          
    
    for session_index, session in enumerate(sessions):
      if clickout_index[session_index] != []:
        categories, categories_scores = evaluate_classification(model, session, hotel_dict, n_features, hotels_window[session_index], max_window,  meta_dict, meta_list, prev_hotel_list[session_index])
        
        df_test.loc[clickout_index[session_index], 'item_recommendations'] = list_to_space_string(categories)
        for hotel_i, hotel in enumerate(categories):
          # Write single hotel score
          file_writer.writerow([str(session[0]['session_id']), str(hotel), str(categories_scores[hotel_i])])
  df_sub = get_submission_target(df_test)
  
  #Removing unnecessary columns
  df_sub = df_sub[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]

  mask = df_sub["item_recommendations"].notnull()
  df_sub = df_sub[mask]

  # Saving df_sub
  if isprint:
      df_sub.to_csv('./' + subname + '.csv')

  if dev:
    mrr = 0
  else:
    mrr = score_submissions_no_csv(df_sub, df_gt, get_reciprocal_ranks)

  return mrr
  
