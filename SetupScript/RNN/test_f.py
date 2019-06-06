import csv
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

def recommendations_from_output(output, hotel_dict, hotels_window, n_features):
    i = 0
    window_dict = {}
    output_arr = np.asarray(output[0].cpu().detach().numpy())
    ranked_hotels = {}
    hotel_i = 0
    
    for hotel_v in output_arr:
        hotel_id = hotel_dict.index2word[hotel_i]

        if hotel_id in hotels_window:
            ranked_hotels[hotel_id] = hotel_v
        hotel_i = hotel_i + 1
    
    for hotel_id in hotels_window:
        if hotel_id not in ranked_hotels:
            ranked_hotels[hotel_id] = 0

    ranked_hotels = sorted(ranked_hotels.items(), key=itemgetter(1))
    ranked = []

    for tup in ranked_hotels:
        ranked.append(tup[0])                          
                            
    return list_to_space_string(ranked)

def evaluate(model, session, hotel_dict, n_features, hotels_window):
    """Just return an output list of hotel given a single session."""
    
    session_tensor = lstm.session_to_tensor(session, hotel_dict, n_features)
    
    output = lstm.model(session_tensor)

    output = recommendations_from_output(output, hotel_dict, hotels_window, n_features)

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

    # score each row
    df_subm_with_key['score'] = df_subm_with_key.apply(objective_function, axis=1)
    mrr = df_subm_with_key.score.mean()

    return mrr
  
def score_submissions(subm_csv, gt_csv, objective_function):
    """Score submissions with given objective function."""

    print(f"Reading ground truth data {gt_csv} ...")
    df_gt = f.read_into_df(gt_csv)

    print(f"Reading submission data {subm_csv} ...")
    df_subm = f.read_into_df(subm_csv)
    print('Submissions')
    print(df_subm.head(10))

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

def test_accuracy(model, df_test, df_gt, hotel_dict, n_features, subname):
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
                    df_test.loc[action_index, 'item_recommendations'] = evaluate(model, temp_session, hotel_dict, n_features, hotels_window)

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
    df_sub.to_csv('./' + subname + '.csv')

    mrr = score_submissions_no_csv(df_sub, df_gt, get_reciprocal_ranks)
    return mrr