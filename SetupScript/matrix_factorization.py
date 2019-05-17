import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f
from lightfm import LightFM
from scipy import sparse
import math
import operator
import collections as cl

def get_rec_matrix(df_train, df_test, **kwargs):
    subm_csv = 'submission_popular.csv'
    # Select only a portion of dataset for testing purpose
    df_train = f.get_interaction_actions(df_train)
    df_test_cleaned = f.get_interaction_actions(df_test, True)
    df_test_cleaned = remove_null_clickout(df_test_cleaned)
    df_train = pd.concat([df_train, df_test_cleaned], ignore_index=True)
    df_test_user, df_test_nation = split_one_action(df_test)
    #df_out_user, df_missed = generate_prediction_per_user(df_train, df_test_user)
    df_out_user = pd.DataFrame()
    df_missed = pd.DataFrame()
    df_out_nation, df_missed = generate_prediction_per_nation(df_train, df_test_nation, df_missed=df_missed)
    print('There are #' + str(df_missed.shape[0]) + ' items with no predictions')
    print(df_missed.head())
    #duplicates = df_out_user['user_id', 'session_id']][df_out_user[].isin(df_out_nation)]
    #print('There are #' + str(duplicates.shape[0]) + ' items with duplicate predictions')
    #print(duplicates.head())
    df_out = pd.concat([df_out_user, df_out_nation], ignore_index=True)
    print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)
    return df_out


def generate_prediction_per_user(df_train, df_test_user):

    """
        Expected input:
            - df_train = df concatenates train + test
            - df_test_user = df without single action clickout to predict
        Expected output:
            - df_out_user = df with recommended items for each user in the df_test_user
            - df_missed 
    """
    print('Start predicting item for user')
    df_interactions = get_n_interaction(df_train)
    user_dict = create_user_dict(df_interactions)
    hotel_dict = create_item_dict(df_interactions)
    interaction_matrix = f.create_sparse_interaction_matrix(df_interactions, user_dict, hotel_dict)
    mf_model = runMF(interactions = interaction_matrix,k = 300, n_components = 10, loss = 'warp-kos', epoch = 2, n_jobs = 4)
    before = df_test_user.shape[0]
    df_test_user = df_test_user[df_test_user['user_id'].isin(list(user_dict.keys()))]
    print("User missed: " + str(before - df_test_user.shape[0]))
    print('Calculate submissions per user')
    df_test_user['item_recommendations'] = df_test_user.apply(lambda x: sample_recommendation_user(mf_model, interaction_matrix, x.impressions, x.user_id, user_dict, hotel_dict), axis=1)
    df_missed = df_test_user[df_test_user['item_recommendations'] == ""]
    df_test_user = df_test_user[df_test_user['item_recommendations'] != ""]
    print('No prediction for #' + str(df_missed.shape[0]) + 'users')
    print(df_missed.head())
    df_out_user = df_test_user[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]

    return df_out_user, df_missed

def generate_prediction_per_nation(df_train, df_test_nation, df_missed = pd.DataFrame()):
    print('Start predicting the single action clickout')
    df_interactions_nations = get_n_interaction_nation(df_train)
    nation_dict = create_user_dict(df_interactions_nations, col_name='platform')
    hotel_dict = create_item_dict(df_interactions_nations)
    interaction_matrix_nation = f.create_sparse_interaction_matrix(df_interactions_nations, nation_dict, hotel_dict, user_col='platform')
    mf_model = runMF(interactions = interaction_matrix_nation,k = 300, n_components = 300, loss = 'warp-kos', epoch = 30, n_jobs = 4)
    print('Add the #' + str(df_missed.shape[0]) + ' items missed before')
    df_test_nation = pd.concat([df_test_nation, df_missed], ignore_index=True)
    print('Calculate submissions per nation')
    df_test_nation['item_recommendations'] = df_test_nation.apply(lambda x: sample_recommendation_user(mf_model, interaction_matrix_nation, x.impressions, x.platform, nation_dict, hotel_dict, complete = True), axis=1)
    df_missed = df_test_nation[df_test_nation['item_recommendations'] == ""]
    print('No prediction for #' + str(df_missed.shape[0]) + 'items')
    df_out_nation = df_test_nation[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]
    return df_out_nation, df_missed

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
    return df_no_single_action, df_single_action;

def get_n_interaction_nation(df):
    """
    Returns a dataframe with:
    nation | item_id | n_interactions
    """
    print('Get number of occurrences for each pair (nation,item)')
    df = df[['platform','reference']]
    df = (
        df
        .groupby(["platform", "reference"])
        .size()
        .reset_index(name="n_interactions")
    )
    print('First elements of the matrix')
    print(df.head())
    return df

def get_n_interaction(df):
    """ 
    Returns a dataframe with:
    user_id | item_id | n_interactions
    """
    print('Get number of occurrences for each pair (user,item)')
    df = df[['user_id','reference']]
    df = (
        df
        .groupby(["user_id", "reference"])
        .size()
        .reset_index(name="n_interactions")
    )
    print('First elements of the matrix')
    print(df.head())
    return df

def remove_single_actions(df):
    df = df.drop(df[(df['action_type'] == "clickout item") & (df['step'] == 1) & (df['reference'].isnull())].index)
    return df

def get_single_actions(df):
    df = df[(df['action_type'] == "clickout item") & (df['step'] == 1) & (df['reference'].isnull())]
    return df

def runMF(interactions, n_components=30, loss='warp', k=15, epoch=30,n_jobs = 4):
    '''
    Function to run matrix-factorization algorithm
    Required Input -
        - interactions = dataset create by create_interaction_matrix
        - n_components = number of embeddings you want to create to define Item and user
        - loss = loss function other options are logistic, brp
        - epoch = number of epochs to run 
        - n_jobs = number of cores used for execution 
    Expected Output  -
        Model - Trained model
    '''
    print('Starting building a model')
    #x = sparse.csr_matrix(interactions.values)
    model = LightFM(no_components= n_components, loss=loss, k=k, learning_schedule='adadelta', learning_rate=0.5)
    model.fit(interactions,epochs=epoch,num_threads = n_jobs)
    return model

def create_user_dict(interactions, col_name='user_id'):
    '''
    Function to create a user dictionary based on their index and number in interaction dataset
    Required Input - 
        interactions - 
    Expected Output -
        user_dict - Dictionary type output containing interaction_index as key and user_id as value
    '''
    user_id = list(interactions[col_name].drop_duplicates())
    user_dict = {}
    counter = 0 
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict
    
def create_item_dict(interactions, col_name='reference'):
    '''
    Function to create a user dictionary based on their index and number in interaction dataset
    Required Input - 
        interactions 
    Expected Output -
        item_dict - Dictionary type output containing interaction_index as key and item_id as value
    '''
    item_id = list(interactions[col_name].drop_duplicates())
    item_dict = {}
    counter = 0 
    for i in item_id:
        item_dict[i] = counter
        counter += 1
    return item_dict


def sample_recommendation_user(model, interactions, impressions, user_id, user_dict, item_dict, complete=False):
    '''
    Function to produce user recommendations
    Required Input - 
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - user_id = user ID for which we need to generate recommendation
        - user_dict = Dictionary type input containing interaction_index as key and user_id as value
    Expected Output - 
        - Space separated string of recommended hotels
    '''
    user_x = user_dict[user_id]
    items_to_predict = impressions.split('|')
    # Create a dictionary with key=item_id, value=score
    item_missed = []
    
    encoded_item = []
    decoded_item = []
    for item in items_to_predict:
        if item in item_dict: # Se ho già interagito con l'hotel -> Predict
            item_x = item_dict[item]
            encoded_item.append(item_x)
            decoded_item.append(item)
        else:
            item_missed.append(item)
    if(len(encoded_item) == 0):
        if(complete):
            hotel_rec = f.list_to_space_string(items_to_predict)
        else:
            hotel_rec = ""
    else:
        scores = model.predict(user_x, encoded_item)
        hotel_dic = dict(zip(decoded_item, scores))
        for i in item_missed: #Set score 0 for the missed hotels -> To improve with another system
            hotel_dic[i] = -10
        sorted_x = sorted(hotel_dic.items(), key=operator.itemgetter(1), reverse = True)
        sorted_items = list(map(lambda x:x[0], sorted_x))
        hotel_rec = f.list_to_space_string(sorted_items)
    
    return hotel_rec
    