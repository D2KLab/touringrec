import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f
from lightfm import LightFM
from scipy import sparse
import math

def get_rec_matrix(df_train, df_test, **kwargs):
    
    # Select only a portion of dataset for testing purpose
    df_train = f.get_interaction_actions(df_train)
    df_test = f.get_interaction_actions(df_test, True)
    df_train = remove_single_actions(df_train)
    df_train = f.get_df_percentage(df_train, 0.02)
    print('Taken #: ' + str(df_train.shape[0]) + ' rows')
    df_test = remove_single_actions(df_test)
    df_test = f.get_df_percentage(df_test, 0.01)
    users = df_test['user_id'].drop_duplicates()
    print(users.head(10))
    print(df_test.head())
    df_train = pd.concat([df_train, df_test], ignore_index=True)
    #df_train['user_id'] = df_train['user_id'].apply(lambda x: convertToNumber(x))
    print(df_train.head())
    df_interactions = get_n_interaction(df_train)
    interaction_matrix = f.create_interaction_matrix(df_interactions,'user_id', 'reference', 'n_interactions')
    print(interaction_matrix.head())
    mf_model = runMF(interactions = interaction_matrix, n_components = 30, loss = 'warp', epoch = 30, n_jobs = 4)
    user_dict = create_user_dict(interactions = interaction_matrix)
    #hotel_dict = create_item_dict(df = df_train, id_col = 'reference', name_col = 'reference')

    rec_list = sample_recommendation_user(model = mf_model, interactions = interaction_matrix, user_id = '1M9Q1ZR5Q5FJ', user_dict = user_dict, threshold = 4, nrec_items = 10, show = True)
    return

''' 
Returns a dataframe with:
user_id | item_id | n_interactions
'''
def get_n_interaction(df):
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
    x = sparse.csr_matrix(interactions.values)
    model = LightFM(no_components= n_components, loss=loss,k=k)
    model.fit(x,epochs=epoch,num_threads = n_jobs)
    return model

def create_user_dict(interactions):
    '''
    Function to create a user dictionary based on their index and number in interaction dataset
    Required Input - 
        interactions - dataset create by create_interaction_matrix
    Expected Output -
        user_dict - Dictionary type output containing interaction_index as key and user_id as value
    '''
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0 
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict
    
def create_item_dict(df,id_col,name_col):
    '''
    Function to create an item dictionary based on their item_id and item name
    Required Input - 
        - df = Pandas dataframe with Item information
        - id_col = Column name containing unique identifier for an item
        - name_col = Column name containing name of the item
    Expected Output -
        item_dict = Dictionary type output containing item_id as key and item_name as value
    '''
    item_dict ={}
    for i in range(df.shape[0]):
        item_dict[(df.loc[i,id_col])] = df.loc[i,name_col]
    return item_dict

def sample_recommendation_user(model, interactions, user_id, user_dict, threshold = 0,nrec_items = 10, show = True):
    '''
    Function to produce user recommendations
    Required Input - 
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - user_id = user ID for which we need to generate recommendation
        - user_dict = Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - threshold = value above which the rating is favorable in new interaction matrix
        - nrec_items = Number of output recommendation needed
    Expected Output - 
        - Prints list of items the given user has already bought
        - Prints list of N recommended items  which user hopefully will be interested in
    '''
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interactions.loc[user_id,:] \
                                 [interactions.loc[user_id,:] > threshold].index) \
								 .sort_values(ascending=False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items))
    scores = list(pd.Series(return_score_list))
    if show == True:
        print("Known Likes:")
        counter = 1
        for i in known_items:
            print(str(counter) + '- ' + i)
            counter+=1

        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
    return return_score_list
    