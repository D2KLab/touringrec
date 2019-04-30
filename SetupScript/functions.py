import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
from datetime import datetime


def get_submission_target(df):
    """Identify target rows with missing click outs."""

    mask = df["reference"].isnull() & (df["action_type"] == "clickout item")
    df_out = df[mask]

    return df_out

def create_interaction_matrix(df,user_col, item_col, rating_col, norm= False, threshold = None):
    '''
    Function to create an interaction matrix dataframe from transactional type interactions
    Required Input -
        - df = Pandas DataFrame containing user-item interactions
        - user_col = column name containing user's identifier
        - item_col = column name containing item's identifier
        - rating col = column name containing user feedback on interaction with a given item
        - norm (optional) = True if a normalization of ratings is needed
        - threshold (required if norm = True) = value above which the rating is favorable
    Expected output - 
        - Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
    '''
    interactions = df.groupby([user_col, item_col])[rating_col] \
            .sum().unstack().reset_index(). \
            fillna(0).set_index(user_col)
    if norm:
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    return interactions


def string_to_array(s):
    """Convert pipe separated string to array."""

    if isinstance(s, str):
        out = s.split("|")
    elif math.isnan(s):
        out = []
    else:
        raise ValueError("Value must be either string of nan")
    return out

def get_all_properties(df_meta):
    d = []
    for properties in df_meta['properties']:
        temp = properties.split('|')
        for property in temp:
            d.append(property)
    prop_dict = set(d)
    return list(prop_dict)

def explode(df_in, col_expl, flag_conversion = True):
    """Explode column col_expl of array type into multiple rows."""

    df = df_in.copy()
    df.loc[:, col_expl] = df[col_expl].apply(string_to_array)

    df_out = pd.DataFrame(
        {col: np.repeat(df[col].values,
                        df[col_expl].str.len())
         for col in df.columns.drop(col_expl)}
    )

    df_out.loc[:, col_expl] = np.concatenate(df[col_expl].values)
    if(flag_conversion):
        df_out.loc[:, col_expl] = df_out[col_expl].apply(int)

    return df_out

    
def read_into_df(file):
    """Read csv file into data frame."""
    df = (
        pd.read_csv(file)
            .set_index(['user_id', 'session_id', 'timestamp', 'step'])
    )

    return df


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


def get_reciprocal_ranks(ps):
    """Calculate reciprocal ranks for recommendations."""
    mask = ps.reference == np.array(ps.item_recommendations)

    if mask.sum() == 1:
        rranks = generate_rranks_range(0, len(ps.item_recommendations))
        return np.array(rranks)[mask].min()
    else:
        return 0.0


def score_submissions(subm_csv, gt_csv, objective_function):
    """Score submissions with given objective function."""

    print(f"Reading ground truth data {gt_csv} ...")
    df_gt = read_into_df(gt_csv)

    print(f"Reading submission data {subm_csv} ...")
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
    
def score_submissions_no_csv(df_subm, df_gt, objective_function):
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

''' Telegram notification handling '''
# https://api.telegram.org/bot804159482:AAGWfE5RiCKbW-92nLJehda3B9v_51yc6cI/getUpdates
def send_telegram_message(message):
    bot_token = '804159482:AAGWfE5RiCKbW-92nLJehda3B9v_51yc6cI' 
    chat_id = '124425954'

    send_text = "https://api.telegram.org/bot" + bot_token + "/sendMessage?chat_id=" + chat_id + "&text=" + message #+ " @ {}"  
    #response = requests.get(send_text.format(datetime.now().strftime("%H:%M:%S")))
    response = requests.get(send_text)
    return response.json()