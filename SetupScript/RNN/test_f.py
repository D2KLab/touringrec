import csv
import numpy as np
import pandas as pd
import LSTM as lstm
import math as math
import operator
from operator import itemgetter


def list_to_space_string(l):
    """Return a space separated string from a list"""
    s = " ".join(l)
    return s
  
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


def prepare_test(df_test, df_gt):
  #Creating a NaN column for item recommendations
  df_test['item_recommendations'] = np.nan

  test_dim = len(df_test)

  temp_session = []
  test_sessions = []

  temp_clickout_index = []
  test_clickout_index = []

  hotels_window = []
  test_hotels_window = []

  i = 0
  step = 0

  #splitting in sessions while evaluating recommendations for NaN clickouts
  for action_index, action in df_test.iterrows():
      if(action['reference'] != 'unknown'):
          if (action['action_type'] == 'clickout item') & math.isnan(float(action['reference'])):
              hotels_window = action['impressions'].split('|')
              temp_session.append(action)
              temp_clickout_index.append(action_index)
          else:
              temp_session.append(action)

      if(i < test_dim-1):
          if action['session_id'] != df_test.iloc[[i + 1]]['session_id'].values[0]:
              step = 0
              test_sessions.append(temp_session)
              test_hotels_window.append(hotels_window)
              test_clickout_index.append(temp_clickout_index)
              temp_session = []
              hotels_window = []
              temp_clickout_index = []


      i = i+1  
      step = step + 1
        
  return test_sessions, test_hotels_window, test_clickout_index
  

### FUNCTIONS FOR CLASSIFICATION TASK ###

def assign_score(hotel, output_arr, hotel_to_index_dict):
  if hotel not in hotel_to_index_dict:
    return (hotel, -999)
  else:
    return (hotel, output_arr[hotel_to_index_dict[hotel]])

def recommendations_from_output_classification(output, hotel_to_index_dict, window, n_features):
  output_arr = np.asarray(output.cpu().detach().numpy())

  category_tuples = list(map(lambda x: assign_score(x, output_arr[0], hotel_to_index_dict), window))
  category_tuples = sorted(category_tuples, key=lambda tup: tup[1], reverse = True)

  # Converting to 2 lists
  category_dlist = list(map(list, zip(*category_tuples)))

  categories = category_dlist[0]
  categories_scores = category_dlist[1]
  
  return categories, categories_scores

# Just return an output given a line
def evaluate_classification(model, session, hotel_dict, hotel_to_index_dict, n_features, hotels_window):
    line_tensor = lstm.session_to_tensor_ultimate(session, hotel_dict, n_features, hotels_window)
    
    output = model(line_tensor)
        
    output = recommendations_from_output_classification(output, hotel_to_index_dict, hotels_window, n_features)

    return output
  
def test_accuracy_optimized_classification(model, df_test, df_gt, session_dict, clickout_dict, impression_dict, hotel_dict, hotel_to_index_dict, n_features, dir, subname="submission_default_name", isprint=False, dev = False):
  """Return the score obtained by the net on the test dataframe"""
  
  #missed_target = 0
  if dev:
    fname = dir + 'rnn_test_sub_xgb_dev' + subname + '.csv'
  else:
    fname = dir + 'rnn_test_sub_xgb_inner' + subname + '.csv'

  with open(fname, mode='w') as test_xgb_sub:
    
    file_writer = csv.writer(test_xgb_sub)
    file_writer.writerow(['session_id', 'hotel_id', 'score'])          
    
    for session, hotel_list in session_dict.items():
      if session in clickout_dict:
        categories, categories_scores = evaluate_classification(model, hotel_list, hotel_dict, hotel_to_index_dict, n_features, impression_dict[session])
        df_test.loc[(df_test['session_id'] == session) & (df_test['step'] == clickout_dict[session]), 'item_recommendations'] = list_to_space_string(categories)

        for hotel_i, hotel in enumerate(categories):
          # Write single hotel score
          file_writer.writerow([str(session), str(hotel), str(categories_scores[hotel_i])])

  df_sub = get_submission_target(df_test)

  #Removing unnecessary columns
  df_sub = df_sub[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]

  mask = df_sub["item_recommendations"].notnull()

  df_sub = df_sub[mask]
  df_sub = df_sub.drop(df_sub.index[df_sub['item_recommendations'] == math.nan])


  # Saving df_sub
  if isprint:
      df_sub.to_csv(dir + subname + '.csv')

  # Computing mrr only if test set is not the one without gt
  #if df_gt == '':
  #  mrr = 0
  #else:
  mrr = score_submissions_no_csv(df_sub, df_gt, get_reciprocal_ranks)

  return mrr
  
