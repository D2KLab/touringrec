import pandas as pd
import numpy as np
import functions as f
from lightfm import LightFM
from scipy import sparse
import math
import operator
import collections as cl
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time
import xgboost as xgb
from numpy import sort

TRAINING_COLS = ['position','recent_index', 'user_bias', 'item_bias', 'lightfm_dot_product', 'lightfm_prediction', 'score']

def get_rec_matrix(df_train, df_test, inner_train, inner_gt, subm_csv, parameters = None, **kwargs):

    hotel_prices_file = kwargs.get('file_metadata', None)
    df_inner_train = pd.read_csv(inner_train)
    df_inner_gt = pd.read_csv(inner_gt)
    df_inner_gt = get_validation_set(df_test)
    df_inner_train = f.get_interaction_actions(df_inner_train, actions = parameters.listactions)
    df_inner_gt = f.get_interaction_actions(df_inner_gt, actions=parameters.listactions)
    df_inner_gt = remove_single_clickout_actions(df_inner_gt)
    df_inner_gt = create_recent_index(df_inner_gt)
    df_inner_gt_clickout, df_inner_gt_no_clickout = split_clickout(df_inner_gt)
    df_test_cleaned = f.get_interaction_actions(df_test, actions = parameters.listactions, clean_null = True)
    df_test_cleaned = f.remove_null_clickout(df_test_cleaned)
    test_interactions = create_recent_index(df_test_cleaned, grouped=True)
    df_train = pd.concat([df_inner_train, df_inner_gt_no_clickout, df_test_cleaned], sort=False)
    user_dict = create_user_dict(df_train)

    if hotel_prices_file != None:
        hotel_features, hotel_dict = get_hotel_prices(hotel_prices_file, df_train)
    else:
        hotel_features = None

    df_test_user, df_test_nation = split_one_action(df_test)
    mf_model = train_mf_model(df_train, parameters, item_features = hotel_features, hotel_dic = hotel_dict, user_dic = user_dict)
    df_train_xg = get_lightFM_features(df_inner_gt_clickout, mf_model, user_dict, hotel_dict, item_f=hotel_features)
    #df_train_xg = get_RNN_features(df_train_xg, 'rnn_test_sub_xgb_inner_100%_vanilla_opt_0,001lr.csv')
    xg_model = xg_boost_training(df_train_xg)
    df_test_xg = get_lightFM_features(df_test_user, mf_model, user_dict, hotel_dict, item_f=hotel_features, is_test = True)
    df_test_xg = (df_test_xg.merge(test_interactions, left_on=['session_id'], right_on=['session_id'], how="left"))
    df_test_xg['recent_index'] = df_test_xg.apply(lambda x : recent_index(x), axis=1)
    del df_test_xg['all_interactions']
    #df_test_xg = get_RNN_features(df_test_xg, 'rnn_test_sub_xgb_dev_100%_vanilla_opt_0,001lr.csv')
    print(df_test_xg.head())
    df_out = generate_submission(df_test_xg, xg_model)
    df_out_nation = complete_prediction(df_test_nation)
    df_out = pd.concat([df_out, df_out_nation])
    df_out.to_csv(subm_csv, index=False)
    return df_out


def get_validation_set(df):
    sessions = get_non_null_clickout(df)
    dft = df[df['session_id'].isin(sessions)]
    dft = dft[~dft['reference'].isnull()]
    return dft

def get_non_null_clickout(df_test):
    print(df_test.head())
    df_clickout = df_test[(~df_test['reference'].isnull()) & (df_test['action_type'] == 'clickout item')]
    return df_clickout['session_id'].drop_duplicates()

def create_recent_index(df_orig, grouped=False):
    df_list_int = df_orig.groupby('session_id').apply(lambda x: get_list_session_interactions(x)).reset_index(name='all_interactions')
    df_list_int = df_list_int[['session_id', 'all_interactions']]
    if(grouped):
        return df_list_int
    df_orig = (df_orig.merge(df_list_int, left_on=['session_id'], right_on=['session_id'], how="left"))
    return df_orig

def recent_index(x):
    #least_recent = len(x.all_interactions)
    list_interactions = x.all_interactions.split(" ")
    if str(x.item_id) in list_interactions:
        i = list_interactions.index(str(x.item_id))
        i = i / len(list_interactions)
    else:
        i = -999
    #i = least_recent - i
    return i

def get_list_session_interactions(group):
    group.loc[:,'reference'] = group['reference'].apply(str)
    list_values = group.reference.drop_duplicates()
    joined = " ".join(list_values)
    return joined
    
def get_RNN_features(df, filename):
    df_rnn = pd.read_csv(filename)
    df_rnn = df_rnn.rename(columns={'hotel_id':'item_id'})
    df = (df.merge(df_rnn, left_on=['session_id', 'item_id'], right_on=['session_id', 'item_id'], how="left", suffixes=('_mf', '_rnn')))
    df.fillna(0)
    print(df.head())
    return df

def generate_submission(df, xg_model, training_cols=TRAINING_COLS):
    df = df.groupby(['user_id', 'session_id', 'timestamp', 'step']).apply(lambda x: calculate_rank(x, xg_model, t_cols=training_cols)).reset_index(name='item_recommendations')
    df = df[['user_id', 'session_id', 'timestamp', 'step', 'item_recommendations']]
    return df

def calculate_rank(group, model, t_cols=TRAINING_COLS):
    #cols = ['user_id', 'session_id', 'timestamp', 'item_id', 'step']
    #print(group)
    df_test = group[t_cols]
    #print(df_test)
    xgtest = xgb.DMatrix(df_test)
    prediction = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
    dic_pred = dict(zip(group['item_id'].apply(str), prediction))
    sorted_x = sorted(dic_pred.items(), key=operator.itemgetter(1), reverse = True)
    sorted_items = list(map(lambda x:x[0], sorted_x))
    #df = group.iloc[0]
    #df['item_recommendations'] = " ".join(sorted_items)
    return " ".join(sorted_items)

def remove_single_clickout_actions(df):
    print('Initial size: ' + str(df.shape[0]))
    n_action_session = df.groupby('session_id').size().reset_index(name='n_actions')
    print(n_action_session.head())
    df = (df.merge(n_action_session, left_on='session_id', right_on='session_id', how="left"))
    print(df.head())
    df = df.drop(df[(df["action_type"] == "clickout item") & (df['n_actions'] == 1)].index)
    del df['n_actions']
    return df

def xg_boost_training(train):
    train = train[TRAINING_COLS + ['label']]
    df_train, df_val = train_test_split(train, test_size=0.2)
    print(df_train.head())
    cols = ['label']
    xgtrain = xgb.DMatrix(df_train.drop(cols, axis=1), df_train.label)
    xgval = xgb.DMatrix(df_val.drop(cols, axis=1), df_val.label)
    params = {
        'objective':'binary:logistic', 
        'eta':0.1, 
        'booster':'gbtree',
        'predictor': 'cpu_predictor',
        'max_depth':7,         
        'nthread':4,  
        'seed':1,    
        'eval_metric':'auc',
    }

    model = xgb.train(
        params=list(params.items()),  
        early_stopping_rounds=30, 
        verbose_eval=10, 
        dtrain=xgtrain,
        evals=[(xgtrain, 'train'), (xgval, 'test')],
        num_boost_round=300,
    )
    return model

def get_lightFM_features(df, mf_model, user_dict, hotel_dict, item_f = None, user_f=None, is_test = False):
    df_train_xg = f.explode_position_scalable(df, 'impressions')
    if(is_test == False):
        df_train_xg['recent_index'] = df_train_xg.apply(lambda x : recent_index(x), axis=1)
        df_train_xg = df_train_xg[['user_id', 'session_id', 'timestamp', 'step', 'reference', 'position', 'item_id', 'recent_index']]
    else:
        df_train_xg = df_train_xg[['user_id', 'session_id', 'timestamp', 'step', 'reference', 'position', 'item_id']]
    
    #df_train_xg = create_recent_index(df_train_xg)
    if(is_test == False):
        df_train_xg['label'] = df_train_xg.apply(lambda x: 1 if (str(x.item_id) == str(x.reference)) else 0, axis=1)
    df_train_xg['user_id_enc'] = df_train_xg['user_id'].map(user_dict)
    df_train_xg['item_id_enc'] = df_train_xg['item_id'].map(hotel_dict)
    df_train_xg_null = df_train_xg[(df_train_xg['item_id_enc'].isnull())]
    df_train_xg_not_null = df_train_xg[~(df_train_xg['item_id_enc'].isnull())]
    #df_train_xg_not_null = df_train_xg[(~df_train_xg['item_id_enc'].isnull()) & (~df_train_xg['user_id_enc'].isnull())]
    #df_train_xg_null = df_train_xg[(df_train_xg['item_id_enc'].isnull()) | (df_train_xg['user_id_enc'].isnull())]
    print('Utenti nulli')
    df_user_null = df_train_xg[df_train_xg['user_id_enc'].isnull()]
    df_user_null = df_user_null['user_id'].drop_duplicates()
    print(df_user_null) 
    print('There are # ' + str(df_train_xg_not_null.shape[0]) + ' not null pairs')
    print('There are # ' + str(df_train_xg_null.shape[0]) + ' null pairs')
    df_train_xg_not_null.loc[:,'user_id_enc'] = df_train_xg_not_null['user_id_enc'].apply(int)
    df_train_xg_not_null.loc[:,'item_id_enc'] = df_train_xg_not_null['item_id_enc'].apply(int)
    #df_train_xg = df_train_xg.fillna('no_data')
    #df_train_xg_cleaned, df_train_xg_errors = split_no_info_hotel(df_train_xg)
    df_train_xg_not_null.loc[:,'score'] = mf_model.predict(np.array(df_train_xg_not_null['user_id_enc']), np.array(df_train_xg_not_null['item_id_enc']), item_features=item_f, num_threads=4)
    df_train_xg_null.loc[:,'score'] = -999
    df_train_xg_not_null.loc[:,'user_bias'] = mf_model.user_biases[df_train_xg_not_null['user_id_enc']]
    df_train_xg_null.loc[:,'user_bias'] = -999
    df_train_xg_not_null.loc[:,'item_bias'] = mf_model.item_biases[df_train_xg_not_null['item_id_enc']]
    df_train_xg_null.loc[:,'item_bias'] = -999
    user_embeddings = mf_model.user_embeddings[df_train_xg_not_null.user_id_enc]
    item_embeddings = mf_model.item_embeddings[df_train_xg_not_null.item_id_enc]
    df_train_xg_not_null.loc[:,'lightfm_dot_product'] = (user_embeddings * item_embeddings).sum(axis=1)
    df_train_xg_null.loc[:,'lightfm_dot_product'] = -999
    df_train_xg_not_null.loc[:,'lightfm_prediction'] = df_train_xg_not_null['lightfm_dot_product'] + df_train_xg_not_null['user_bias'] + df_train_xg_not_null['item_bias']
    df_train_xg_null.loc[:,'lightfm_prediction'] = -999
    df_train_xg = pd.concat([df_train_xg_not_null, df_train_xg_null], ignore_index=True, sort=False)
    df_train_xg = df_train_xg.sort_values(by=['user_id', 'session_id', 'timestamp', 'step'], ascending=False)
    cols = ['reference', 'user_id_enc', 'item_id_enc']
    df_train_xg = df_train_xg.drop(cols, axis=1)
    return df_train_xg

def split_clickout(df):
    print('Size iniziale: ' + str(df.shape[0]))
    df_clickout = get_clickouts(df)
    df_no_clickout = get_no_clickout(df)
    print('There are #: ' + str(df_clickout.shape[0]) + ' clickout actions')
    print('There are #: ' + str(df_no_clickout.shape[0]) + ' no clickout actions')
    return df_clickout, df_no_clickout

def get_clickouts(df_test):
    df_test['step_max'] = df_test.groupby(['user_id'])['step'].transform(max)
    df_clickout = df_test[(df_test['step_max'] == df_test['step']) & (df_test['action_type'] == 'clickout item')]
    del df_clickout['step_max']
    return df_clickout

def get_no_clickout(df):
    df['step_max'] = df.groupby(['user_id'])['step'].transform(max)
    df_no_clickout = df[(~(df['step_max'] == df['step'])) | (~(df['action_type'] == 'clickout item'))]
    del df_no_clickout['step_max']
    return df_no_clickout

def complete_prediction(df_test_nation):
    df_test_nation['item_recommendations'] = df_test_nation.apply(lambda x: fill_recs(x.impressions), axis=1)
    df_out_nation = df_test_nation[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]
    return df_out_nation

def fill_recs(imp):
    l = imp.split('|')
    return f.list_to_space_string(l)

def train_mf_model(df_train, params, item_features = None, user_features = None, hotel_dic = None, user_dic = None):

    df_interactions = get_n_interaction(df_train, weight_dic = params.actionsweights)
    if user_dic == None:
        print('Null user dictionary. Creating it...')
        user_dic = create_user_dict(df_interactions)
    if hotel_dic == None:
        print('Null hotel dictionary. Creating it...')
        hotel_dic = create_item_dict(df_interactions)
    interaction_matrix = f.create_sparse_interaction_matrix(df_interactions, user_dic, hotel_dic)
    mf_model = runMF(interaction_matrix, params, n_jobs = 4, item_f = item_features, user_f = user_features)
    return mf_model

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
    df_metadata = pd.read_csv(metadata_file)
    df_metadata['price'] = df_metadata['price'].apply(lambda x: math.log10(x))
    max_price = df_metadata['price'].max()
    min_price = df_metadata['price'].min()
    range = (max_price - min_price) / n_categories
    df_metadata['intervals'] = pd.cut(df_metadata['price'], bins=np.arange(min_price,max_price,range))
    df_metadata.loc[:, 'intervals'] = df_metadata['intervals'].apply(str)
    price_dic = pd.Series(df_metadata.intervals.values,index=df_metadata.impressions).to_dict()
    interactions = get_n_interaction(interactions)
    interactions.loc[:, 'reference'] = interactions['reference'].apply(int)
    interactions.loc[:, 'feature'] = interactions['reference'].apply(lambda x : price_dic.get(x))
    interactions['feature'] = interactions['feature'].fillna('no_cat')
    s_matrix, hotel_dic = generate_prices_sparse_matrix(interactions)
    return s_matrix, hotel_dic

def split_one_action(df_test):
    """
    Required Input -
        - df_test = test set dataframe
    Expected Output  -
        - df_no_single_action = dataframe without clickout at step 1 to predict
        - df_single_action = dataframe with only clickout at step 1 to predict
    """
    df_int = f.get_interaction_actions(df_test)
    n_action_session = df_int.groupby('session_id').size().reset_index(name='n_actions')
    df_test = f.get_submission_target(df_test)
    df_test = (df_test.merge(n_action_session, left_on='session_id', right_on='session_id', how="left"))
    print(df_test.head())
    df_no_single_action = remove_single_actions(df_test)
    df_single_action = get_single_actions(df_test)
    print('Total item of test set: ' + str(df_test.shape[0]) + ' No single action: #' + str(df_no_single_action.shape[0]) + ' Only single actions: #' + str(df_single_action.shape[0]))
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
    if(weight_dic == None):
        df = df[[user_col,'reference']]
        df = (
            df
            .groupby([user_col, "reference"])
            .size()
            .reset_index(name="n_interactions")
        )
    else:
        df = df.replace({'action_type': weight_dic})
        df = df[[user_col,'reference', 'action_type']]
        df = df.groupby([user_col, "reference"])['action_type'].agg('sum').reset_index(name='n_interactions')
    return df

def remove_single_actions(df):
    df = df.drop(df[(df['action_type'] == "clickout item") & (df['n_actions'] == 1) & (df['reference'].isnull())].index)
    return df

def get_single_actions(df):
    df = df[(df['action_type'] == "clickout item") & (df['n_actions'] == 1) & (df['reference'].isnull())]
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
    model = LightFM(no_components= params.ncomponents, loss=params.lossfunction, k=params.mfk, learning_schedule=params.learningschedule, learning_rate=params.learningrate, user_alpha=1e-6, item_alpha=1e-6)
    model.fit(interactions,epochs=params.epochs,num_threads = n_jobs, item_features=item_f, user_features=user_f)
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