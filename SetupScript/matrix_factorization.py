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
    df_train = remove_single_actions(df_train)
    #df_train = f.get_df_percentage(df_train, 0.0001)
    print('Train: Taken #: ' + str(df_train.shape[0]) + ' rows')
    df_test_cleaned = remove_single_actions(df_test_cleaned)
    #df_test_cleaned = f.get_df_percentage(df_test_cleaned, 0.15)
    print('Test: Taken #: ' + str(df_test_cleaned.shape[0]) + ' rows')
    df_train = pd.concat([df_train, df_test_cleaned], ignore_index=True)
    print(df_train.head())
    df_interactions = get_n_interaction(df_train)
    user_dict = create_user_dict(df_interactions)
    hotel_dict = create_item_dict(df_interactions)
    interaction_matrix = f.create_sparse_interaction_matrix(df_interactions, user_dict, hotel_dict)
    mf_model = runMF(interactions = interaction_matrix,k = 500, n_components = 200, loss = 'warp-kos', epoch = 100, n_jobs = 4)
    print(df_test.shape[0])
    df_test = f.get_submission_target(df_test)
    print("Size before: " + str(df_test.shape[0]))
    df_test = df_test[df_test['user_id'].isin(list(user_dict.keys()))]
    print("Size after: " + str(df_test.shape[0]))
    #print('Value error #:' + str(verror))
    print('Submissions')
    print(df_test.shape[0])
    print('Calculate submissions')
    df_test['item_recommendations'] = df_test.apply(lambda x: sample_recommendation_user(mf_model, interaction_matrix, x.impressions, x.user_id, user_dict, hotel_dict), axis=1)
    df_test = df_test[df_test['item_recommendations'] != ""]
    df_out = df_test[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]
    print(df_out.head())
    #rec_list = sample_recommendation_user(model = mf_model, interactions = interaction_matrix, user_id = '1M9Q1ZR5Q5FJ', user_dict = user_dict, threshold = 4, nrec_items = 10, show = True)
    print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)

    return df_out


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
    df = df.drop(df[(df['action_type'] == "clickout item") & (df['step'] == 1)].index)
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


def sample_recommendation_user(model, interactions, impressions, user_id, user_dict, item_dict):
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
    