import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f
from lightfm import LightFM
from scipy import sparse
import math
import operator
import collections as cl
from scipy.sparse import csr_matrix
import dataset_clean as dsc
from sklearn.preprocessing import OneHotEncoder
import time
from lightfm.evaluation import reciprocal_rank
    
def get_rec_matrix(df_train, df_test, parameters = None, **kwargs):

    hotel_prices_file = kwargs.get('file_metadata', None)
    subm_csv = 'submission_matrixfactorization.csv'


    # Clean the dataset
    df_train = f.get_interaction_actions(df_train, actions = parameters.listactions)
    df_test_cleaned = f.get_interaction_actions(df_test, actions = parameters.listactions, clean_null = True)
    df_test_cleaned = remove_null_clickout(df_test_cleaned)
    df_train = pd.concat([df_train, df_test_cleaned], ignore_index=True)
    df_test_user = f.get_submission_target(df_test)
    
    u_features, user_dict, nation_dict = generate_user_features(df_train)
    df_test_user, df_test_nation = split_one_action(df_test)

    if hotel_prices_file != None:
        hotel_features, hotel_dict = get_hotel_prices(hotel_prices_file, df_train)
    else:
        hotel_features = None

    df_out_user, df_missed = generate_prediction_per_user(df_train, df_test_user, parameters, item_features = hotel_features, user_features = u_features, nation_dic = nation_dict, hotel_dic = hotel_dict, user_dic = user_dict)
    #df_missed = pd.DataFrame()
    #df_out_user = pd.DataFrame()
    print('There are #' + str(df_missed.shape[0]) + ' items with no predictions')
    #print(df_missed.head())
    df_out_nation = complete_prediction(df_test_nation, df_missed)
    #df_out_nation, df_missed = generate_prediction_per_nation(df_train, df_test_nation, df_missed=df_missed, epochs = epochs, n_comp = ncomponents, lossf = lossfunction, mfk = mfk, item_features = hotel_features, nation_dict = nation_dict, hotel_dict = hotel_dict)
    #print('There are #' + str(df_missed.shape[0]) + ' items with no predictions')
    #print(df_missed.head())

    df_out = pd.concat([df_out_user, df_out_nation], ignore_index=True)
    #print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)
    return df_out


def complete_prediction(df_test_nation, df_missed):
    #print('Start predicting the single action clickout')
    #print('Add the #' + str(df_missed.shape[0]) + ' items missed before')
    df_test_nation = pd.concat([df_test_nation, df_missed], ignore_index=True, sort=False)
    #print('Fill submissions')
    df_test_nation['item_recommendations'] = df_test_nation.apply(lambda x: fill_recs(x.impressions), axis=1)
    df_missed = df_test_nation[df_test_nation['item_recommendations'] == ""]
    #print('No prediction for #' + str(df_missed.shape[0]) + 'items')
    df_out_nation = df_test_nation[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]
    return df_out_nation

def fill_recs(imp):
    l = imp.split('|')
    return f.list_to_space_string(l)

def generate_user_features(df):
    df['present'] = 1
    starting_user = df.shape[0]
    df_user_features = df.drop_duplicates(subset='user_id', keep='first')
    #print('# of duplicates: ' + str(starting_user - df_user_features.shape[0]))
    df_user_features = df_user_features[['user_id', 'platform', 'present']]
    #print('First elements of the dataset')
    #print(df_user_features.head())
    #print('Create dictionaries')
    user_dict = create_user_dict(df_user_features) #Controllare che sia uguale all'altro dizionario
    nation_dict = create_item_dict(df_user_features, col_name='platform')

    list_user = list(df_user_features['user_id'])
    list_nations = list(df_user_features['platform'])
    list_data = list(df_user_features['present'])
    n_user = len(user_dict)
    n_nations = len(nation_dict)
    # Convert each list of string in a list of indexes
    list_user = list(map(lambda x: user_dict[x], list_user))
    list_nations = list(map(lambda x: nation_dict[x], list_nations))
    # Generate the sparse matrix
    row = np.array(list_user)
    col = np.array(list_nations)
    data = np.array(list_data)
    csr = csr_matrix((data, (row, col)), shape=(n_user, n_nations))
    #print(csr.toarray())

    return csr, user_dict, nation_dict



def generate_prediction_per_user(df_train, df_test_user, params, item_features = None, user_features = None, nation_dic = None, hotel_dic = None, user_dic = None):

    """
        Expected input:
            - df_train = df concatenates train + test
            - df_test_user = df without single action clickout to predict
        Expected output:
            - df_out_user = df with recommended items for each user in the df_test_user
            - df_missed 
    """
    #print('Start predicting item for user')
    #Create user dictionary
    df_interactions = get_n_interaction(df_train, weight_dic = params.actionsweights)
    if user_dic == None:
        user_dic = create_user_dict(df_interactions)
    if hotel_dic == None:
        hotel_dic = create_item_dict(df_interactions)
    interaction_matrix = f.create_sparse_interaction_matrix(df_interactions, user_dic, hotel_dic)
    mf_model = runMF(interaction_matrix, params, n_jobs = 4, item_f = item_features, user_f = user_features)
    """
    for tag in (u'AU', u'BR', u'GB'):
        tag_id = nation_dic.get(tag)
        similars = get_similar_tags(mf_model, tag_id)
        print('Similar tag for ' + tag)
        for tid in similars:
            # Iterating over values 
            for nat, id in nation_dic.items(): 
                if (id == tid):
                    print(nat, ":", id) 
    """
    before = df_test_user.shape[0]
    df_test_user = df_test_user[df_test_user['user_id'].isin(list(user_dic.keys()))]
    #print("User missed: " + str(before - df_test_user.shape[0]))
    #print('Calculate submissions per user')
    df_test_user['item_recommendations'] = df_test_user.apply(lambda x: sample_recommendation_user(mf_model, interaction_matrix, x.impressions, x.user_id, user_dic, hotel_dic, hotel_features=item_features), axis=1)
    df_missed = df_test_user[df_test_user['item_recommendations'] == ""]
    df_test_user = df_test_user[df_test_user['item_recommendations'] != ""]
    #print('No prediction for #' + str(df_missed.shape[0]) + 'users')
    #print(df_missed.head())
    df_out_user = df_test_user[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]

    return df_out_user, df_missed

"""
def generate_prediction_per_nation(df_train, df_test_nation, df_missed = pd.DataFrame(), epochs = 30, n_comp = 10, lossf = 'warp-kos', mfk = 200, action_weights = None, item_features = None, nation_dict = None, hotel_dict = None):
    print('Start predicting the single action clickout')
    df_interactions_nations = get_n_interaction(df_train, user_col='platform', weight_dic = action_weights)
    interaction_matrix_nation = f.create_sparse_interaction_matrix(df_interactions_nations, nation_dict, hotel_dict, user_col='platform')
    mf_model = runMF(interactions = interaction_matrix_nation,k = mfk, n_components = n_comp, loss = lossf, epoch = epochs, n_jobs = 4, item_f = item_features)
    print('Add the #' + str(df_missed.shape[0]) + ' items missed before')
    df_test_nation = pd.concat([df_test_nation, df_missed], ignore_index=True)
    print('Calculate submissions per nation')
    df_test_nation['item_recommendations'] = df_test_nation.apply(lambda x: sample_recommendation_user(mf_model, interaction_matrix_nation, x.impressions, x.platform, nation_dict, hotel_dict, complete = True, hotel_features=item_features), axis=1)
    df_missed = df_test_nation[df_test_nation['item_recommendations'] == ""]
    print('No prediction for #' + str(df_missed.shape[0]) + 'items')
    df_out_nation = df_test_nation[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]
    return df_out_nation, df_missed
"""
def get_similar_tags(model, tag_id):
    # Define similarity as the cosine of the angle
    # between the tag latent vectors

    # Normalize the vectors to unit length
    tag_embeddings = (model.user_embeddings.T
                      / np.linalg.norm(model.user_embeddings, axis=1)).T

    query_embedding = tag_embeddings[tag_id]
    similarity = np.dot(tag_embeddings, query_embedding)
    most_similar = np.argsort(-similarity)[1:4]

    return most_similar


def remove_null_clickout(df):
    """
    Remove all the occurences where the clickout reference is set to null (Item to predict)
    """
    df = df.drop(df[(df['action_type'] == "clickout item") & (df['reference'].isnull())].index)
    return df

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
def get_hotel_prices(metadata_file, interactions, n_categories = 2000):
    """
    Required Input -
        - metadata_file = file with the average price for each hotel
    """
    #print("Reading metadata: " + metadata_file)
    df_metadata = pd.read_csv(metadata_file)
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
    """
    items = list(interactions['reference'].drop_duplicates().apply(int).values)
    item_cat = map(price_dic.get, items)

    list_items_category = np.array(list(item_cat))
    """
    interactions = get_n_interaction(interactions)
    #print('# Of items: ' + str(interactions.shape[0]))
    interactions.loc[:, 'reference'] = interactions['reference'].apply(int)
    interactions.loc[:, 'feature'] = interactions['reference'].apply(lambda x : price_dic.get(x))
    interactions['feature'] = interactions['feature'].fillna('no_cat')
    s_matrix, hotel_dic = generate_prices_sparse_matrix(interactions)

    """
    list_items_category = np.array(list(interactions['reference'].values))
    rows = np.arange(len(list_items_category))
    cols = np.repeat(0, len(list_items_category))
    data = np.array(list_items_category).astype('int64')
    ifeatures = csr_matrix((data, (rows, cols)), shape=(len(list_items_category), 1))
    """

    return s_matrix, hotel_dic




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
    #print('Total item of test set: ' + str(df_test.shape[0]) + ' No single action: #' + str(df_no_single_action.shape[0]) + ' Only single actions: #' + str(df_single_action.shape[0]))
    return df_no_single_action, df_single_action


def get_n_interaction(df, user_col='user_id', weight_dic = None):
    """ 
    Returns a dataframe with:
    user_id | item_id | n_interactions
    - Input:
        df -> pandas dataframe
        user_col -> name of the user column
        weight_dic -> weight for each type of interaction
    """
    # If no weight is specified -> All actions have the same weight
    #print('Get number of occurrences for each pair (user,item)')
    if(weight_dic == None):
        #print(df.head())
        df = df[[user_col,'reference']]
        df = (
            df
            .groupby([user_col, "reference"])
            .size()
            .reset_index(name="n_interactions")
        )
        #print(df.head())
    else:
        df = df.replace({'action_type': weight_dic})
        #df['n_interactions'] = df.apply(lambda x : weight_dic[x.action_type], axis=1)
        df = df[[user_col,'reference', 'action_type']]
        df = df.groupby([user_col, "reference"])['action_type'].agg('sum').reset_index(name='n_interactions')

    #print('First elements of the matrix')
    #print(df.head())
    return df

def encode_actions(action, dic):
    value = dic[action]
    return value
    

def remove_single_actions(df):
    df = df.drop(df[(df['action_type'] == "clickout item") & (df['step'] == 1) & (df['reference'].isnull())].index)
    return df

def get_single_actions(df):
    df = df[(df['action_type'] == "clickout item") & (df['step'] == 1) & (df['reference'].isnull())]
    return df

def runMF(interactions, params, n_jobs = 4, item_f = None, user_f = None):
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
    model = LightFM(no_components= params.ncomponents, loss=params.lossfunction, k=params.mfk, learning_schedule=params.learningschedule, learning_rate=params.learningrate, user_alpha=1e-6, item_alpha=1e-6)
    model.fit(interactions,epochs=params.epochs,num_threads = n_jobs, item_features=item_f)
    return model

def runMF_loss(interactions, test_interactions, n_components=30, loss='warp', k=15, epochs=30, n_jobs = 4, item_f = None):
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
    warp_duration = []
    warp_mrr = []
    model = LightFM(no_components= n_components, loss=loss, k=k, learning_schedule='adadelta', learning_rate=0.5)
    for epoch in range(epochs):
        start = time.time()
        model.fit_partial(interactions, epochs=1, num_threads = n_jobs)
        warp_duration.append(time.time() - start)
        print('Start calculating the score for the epoch')
        mrr = reciprocal_rank(model, test_interactions, train_interactions=interactions, num_threads=n_jobs).mean()
        print('Epoch #' + str(epoch) + ' MRR: ' + str(mrr))
        warp_mrr.append(mrr)
        

    x = np.arange(epochs)
    plt.plot(x, np.array(warp_mrr))
    plt.legend(['WARP MRR'], loc='upper right')
    plt.show()

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


def sample_recommendation_user(model, interactions, impressions, user_id, user_dict, item_dict, hotel_features = None, user_nations = None, complete=False):
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
    items_to_predict = list(map(int, items_to_predict))
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
            complete_prediction = list(map(str, items_to_predict))
            hotel_rec = f.list_to_space_string(complete_prediction)
        else:
            hotel_rec = ""
    else:
        scores = model.predict(user_x, encoded_item, item_features=hotel_features)
        hotel_dic = dict(zip(decoded_item, scores))
        sorted_x = sorted(hotel_dic.items(), key=operator.itemgetter(1), reverse = True)
        sorted_items = list(map(lambda x:x[0], sorted_x))
        sorted_items = sorted_items + item_missed
        sorted_items = list(map(str, sorted_items))
        hotel_rec = f.list_to_space_string(sorted_items)
    
    return hotel_rec
    