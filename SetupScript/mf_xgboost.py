import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler   
from numpy import sort
from sklearn.feature_selection import SelectFromModel
#import graphviz
TRAINING_COLS = ['position','recent_index', 'user_bias' , 'item_bias', 'lightfm_dot_product', 'lightfm_prediction', 'score_mf', 'score_gru', 'score_knn', 'score_rule']

def get_rec_matrix(df_train, df_test, parameters = None, **kwargs):

    hotel_prices_file = kwargs.get('file_metadata', None)
    #df_inner_train = pd.read_csv('train_inner.csv')
    df_inner_train = df_train
    #df_inner_gt = pd.read_csv('gt_inner.csv')
    df_inner_gt = get_validation_set(df_test)
    subm_csv = 'submission_mf_xgboost.csv'
    df_train = clean_dataset_error(df_train)
    df_inner_gt = clean_dataset_error(df_inner_gt)
    df_inner_train = clean_dataset_error(df_inner_train)
    print(df_train.head())
    df_test = clean_dataset_error(df_test)
    # Clean the dataset
    df_inner_train = f.get_interaction_actions(df_inner_train, actions = parameters.listactions)
    print(df_inner_train.head())
    #df_train = f.get_interaction_actions(df_train, actions = parameters.listactions)
    df_inner_gt = f.get_interaction_actions(df_inner_gt, actions=parameters.listactions)
    #df_inner_single_click = get_single_clickout_actions(df_inner_gt)
    df_inner_gt = remove_single_clickout_actions(df_inner_gt)
    #print(df_inner_single_click.head())
    df_inner_gt = create_recent_index(df_inner_gt)
    print(df_inner_gt.head())
    #df_train_xg_single = get_single_click_features(df_inner_single_click)
    #xg_model_single_click = xg_boost_training_single_click(df_train_xg_single)
    df_inner_gt_clickout, df_inner_gt_no_clickout = split_clickout(df_inner_gt)

    df_test_cleaned = f.get_interaction_actions(df_test, actions = parameters.listactions, clean_null = True)
    df_test_cleaned = f.remove_null_clickout(df_test_cleaned)
    test_interactions = create_recent_index(df_test_cleaned, grouped=True)
    print(test_interactions.head())
    #dic_pop = f.get_popularity_dictionary(pd.concat([df_train, df_test_cleaned], ignore_index=True, sort=False))
    df_train = pd.concat([df_inner_train, df_inner_gt_no_clickout, df_test_cleaned], sort=False)
    user_dict = create_user_dict(df_train)

    if hotel_prices_file != None:
        hotel_features, hotel_dict = get_hotel_prices(hotel_prices_file, df_train)
    else:
        hotel_features = None

    #u_features = generate_user_features(df_train, user_dict)
    print(df_test.head())
    df_test_user, df_test_nation = split_one_action(df_test)
    print(df_test_user.head())
    mf_model = train_mf_model(df_train, parameters, item_features = hotel_features, hotel_dic = hotel_dict, user_dic = user_dict)
    print('Get training set for XGBoost')
    df_train_xg = get_lightFM_features(df_inner_gt_clickout, mf_model, user_dict, hotel_dict, item_f=hotel_features)
    #df_train_xg = get_FR_xgboost(df_train_xg)
    #df_train_xg = get_RNN_features(df_train_xg, 'rnn_test_sub_xgb_inner.csv')
    print('LightFM Features: ')
    print(df_train_xg.head())
    #df_train_xg['popularity'] = df_train_xg.apply(lambda x : add_popularity(x.item_id, dic_pop), axis=1)
    #df_train_xg = get_most_popular_ranking(df_train_xg, sub_filename='submission_basesolution_nation_inner.csv')
    #df_train_xg = df_train_xg.groupby(['user_id', 'session_id', 'timestamp', 'step']).apply(lambda x: add_popularity(x.item, dic_pop)).reset_index(name='item_recommendations')
    print('Aggiunta della popolarita')
    print(df_train_xg.head())
    print('Generate model for XGBoost')
    xg_model = xg_boost_training(df_train_xg)
    print('Get feature for test set')
    df_test_xg = get_lightFM_features(df_test_user, mf_model, user_dict, hotel_dict, item_f=hotel_features, is_test = True)
    df_test_xg = (df_test_xg.merge(test_interactions, left_on=['session_id'], right_on=['session_id'], how="left"))
    df_test_xg['recent_index'] = df_test_xg.apply(lambda x : recent_index(x), axis=1)
    del df_test_xg['all_interactions']
    #df_test_xg = get_FR_final(df_test_xg)
    #df_test_xg = get_RNN_features(df_test_xg, 'rnn_test_sub_xgb_dev.csv')
    #df_test_xg['popularity'] = df_test_xg.apply(lambda x : add_popularity(x.item_id, dic_pop), axis=1)
    #df_train_xg = get_most_popular_ranking(df_train_xg, sub_filename='submission_basesolution_nation.csv')
    print(df_test_xg.head())
    df_out = generate_submission(df_test_xg, xg_model)
    print('Generated submissions: ')
    print(df_out.head())
    #df_test_nation = f.explode_position_scalable(df_test_nation, 'impressions')
    #df_test_nation['device'] = df_test_nation.apply(lambda x: 0 if(str(x.device) == 'desktop') else 1, axis = 1)
    #print(df_test_nation.head())
    #df_out_nation = generate_submission(df_test_nation, xg_model_single_click, ['position', 'device'])
    df_out_nation = complete_prediction(df_test_nation)
    df_out = pd.concat([df_out, df_out_nation])
    df_out.to_csv(subm_csv, index=False)
    return df_out

def get_single_click_features(df, is_test = False):
    df_train_xg = f.explode_position_scalable(df, 'impressions')
    if(is_test == False):
        df_train_xg['label'] = df_train_xg.apply(lambda x: 1 if (str(x.item_id) == str(x.reference)) else 0, axis=1)
    df_train_xg['device'] = df_train_xg.apply(lambda x: 0 if(str(x.device) == 'desktop') else 1, axis = 1)
    print(df_train_xg.head())
    return df_train_xg

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
    # distinct_hotel = group.reference.drop_duplicates().values
    # dict = {}
    # counter = 0
    # for x in distinct_hotel:
    #     dict[x] = counter
    #     counter += 1
    df_list_int = df_orig.groupby('session_id').apply(lambda x: get_list_session_interactions(x)).reset_index(name='all_interactions')
    df_list_int = df_list_int[['session_id', 'all_interactions']]
    if(grouped):
        return df_list_int
    df_orig = (df_orig.merge(df_list_int, left_on=['session_id'], right_on=['session_id'], how="left"))
    #del df_orig['all_interactions']
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

def add_popularity(item, dictionary):
    if item in dictionary:
        pop = dictionary.get(item)
    else:
        pop = -999
    return pop
    
def get_RNN_features(df, filename):
    df_rnn = pd.read_csv(filename)
    df_rnn = df_rnn.rename(columns={'hotel_id':'item_id'})
    df = (df.merge(df_rnn, left_on=['session_id', 'item_id'], right_on=['session_id', 'item_id'], how="left", suffixes=('_mf', '_gru')))
    df.fillna(0)
    print(df.head())
    return df

def get_FR_xgboost(df):
    print('DF INIZIALE: ' + str(df.shape[0]))
    MERGE_COLS = ['user_id', 'session_id', 'hotel_id', 'timestamp', 'step']
    df_gru = pd.read_csv('GRU_test_dev.csv')
    df_knn = pd.read_csv('KNN_test_dev.csv')
    df_rule = pd.read_csv('Rule_based_test_dev.csv')
    df_gru = (df_gru.merge(df_knn, left_on=MERGE_COLS, right_on=MERGE_COLS, how="left", suffixes=('_gru', '_knn')))
    df_final = (df_gru.merge(df_rule, left_on=MERGE_COLS, right_on=MERGE_COLS, how="left"))
    df_final = clean_FR_dataset(df_final)    
    MERGE_COLS = ['user_id', 'session_id', 'item_id', 'timestamp', 'step']
    df = (df.merge(df_final, left_on=MERGE_COLS, right_on=MERGE_COLS, how="left"))
    df.fillna(0)
    print('DF FINALE: ' + str(df.shape[0]))
    print(df.head())
    return df

def get_FR_final(df):
    print('DF INIZIALE: ' + str(df.shape[0]))
    MERGE_COLS = ['user_id', 'session_id', 'hotel_id', 'timestamp', 'step']
    df_gru = pd.read_csv('GRU_confirmation.csv')
    df_knn = pd.read_csv('KNN_confirmation.csv')
    df_rule = pd.read_csv('Rule_based_confirmation.csv')
    df_gru = (df_gru.merge(df_knn, left_on=MERGE_COLS, right_on=MERGE_COLS, how="left", suffixes=('_gru', '_knn')))
    df_final = (df_gru.merge(df_rule, left_on=MERGE_COLS, right_on=MERGE_COLS, how="left"))
    df_final = clean_FR_dataset(df_final)    
    MERGE_COLS = ['user_id', 'session_id', 'item_id', 'timestamp', 'step']
    df = (df.merge(df_final, left_on=MERGE_COLS, right_on=MERGE_COLS, how="left"))
    df.fillna(0)
    print('DF FINALE: ' + str(df.shape[0]))
    print(df.head())
    return df


def clean_FR_dataset(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.rename(columns={'hotel_id':'item_id'})
    return df



def generate_submission(df, xg_model, training_cols=TRAINING_COLS):
    df = df.groupby(['user_id', 'session_id', 'timestamp', 'step']).apply(lambda x: calculate_rank(x, xg_model, t_cols=training_cols)).reset_index(name='item_recommendations')
    df = df[['user_id', 'session_id', 'timestamp', 'step', 'item_recommendations']]
    return df

def calculate_rank(group, model, t_cols=TRAINING_COLS):
    #cols = ['user_id', 'session_id', 'timestamp', 'item_id', 'step']
    #print(group)
    df_test = group[t_cols]
    #df_test = scale_column(df_test, ['position'])
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
    # df['step_max'] = df.groupby(['session_id'])['step'].transform(max)
    # print(df.head())
    df = df.drop(df[(df["action_type"] == "clickout item") & (df['n_actions'] == 1)].index)
    #print('Initial size: ' + str(df.shape[0]))
    #df = df[df['n_actions'] < 100]
    #print('Without outliers size: ' + str(df.shape[0]))
    del df['n_actions']
    return df

def get_single_clickout_actions(df):
    print('Initial size: ' + str(df.shape[0]))
    n_action_session = df.groupby('session_id').size().reset_index(name='n_actions')
    print(n_action_session.head())
    df = (df.merge(n_action_session, left_on='session_id', right_on='session_id', how="left"))
    print(df.head())
    # df['step_max'] = df.groupby(['session_id'])['step'].transform(max)
    # print(df.head())
    df = df[(df["action_type"] == "clickout item") & (df['n_actions'] == 1)]
    print('Final size: ' + str(df.shape[0]))
    del df['n_actions']
    return df

def clean_dataset_error(df):
    df = df[df['user_id'] != '3473ULL51OOW']
    return df
def xg_boost_training_single_click(train):
    train = train[['position', 'device', 'label']]
    df_train, df_val = train_test_split(train, test_size=0.2)
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

    #xgb.plot_importance(model)
    #xgb.plot_tree(model)
    plt.savefig('importance_xgboost.png')
    #plt.show()
    return model

def scale_column(df, col_name):
    scaled_features = df.copy()
    features = scaled_features[col_name]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    scaled_features[col_name] = features
    return scaled_features


def xg_boost_training(train):
    train = train[TRAINING_COLS + ['label']]
    #train = scale_column(train, ['position'])
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

    xgb.plot_importance(model)
    plt.savefig('importance_xgboost.png')
    #xgb.plot_tree(model)
    #plt.show()
    return model


def get_lightFM_features(df, mf_model, user_dict, hotel_dict, item_f = None, user_f=None, is_test = False):
    df_train_xg = f.explode_position_scalable(df, 'impressions')
    if(is_test == False):
        df_train_xg['recent_index'] = df_train_xg.apply(lambda x : recent_index(x), axis=1)
        df_train_xg = df_train_xg[['user_id', 'session_id', 'timestamp', 'step', 'reference', 'position', 'item_id', 'recent_index']]
    else:
        df_train_xg = df_train_xg[['user_id', 'session_id', 'timestamp', 'step', 'reference', 'position', 'item_id']]
    
    #df_train_xg = create_recent_index(df_train_xg)
    print(df_train_xg.head(500))
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


def get_hotel_position(reference, impressions):
    list_impressions = impressions.split('|')
    return list_impressions.index(reference)

def get_most_popular_ranking(df, sub_filename='submission_basesolution_nation.csv'):
    session_list = df['session_id'].drop_duplicates()
    df_sub_nation = pd.read_csv(sub_filename)
    print(df_sub_nation.head())
    print(str(df_sub_nation.shape[0]))
    df_sub_nation = df_sub_nation[df_sub_nation['session_id'].isin(session_list.values)]
    print(str(df_sub_nation.shape[0]))
    df_sub_nation = f.explode_position_scalable(df_sub_nation, 'item_recommendations', pipe=' ')
    df_sub_nation = df_sub_nation.rename(columns={'position':'mp_rank'})
    print(df_sub_nation.head())
    df = (df.merge(df_sub_nation, left_on=['user_id', 'session_id', 'item_id', 'timestamp', 'step'], right_on=['user_id', 'session_id', 'item_id', 'timestamp', 'step'], how="left"))
    df = df.fillna(-999)
    print(df.head())
    return df_sub_nation

def get_lfm_features(row, user_d, item_d, model, item_f = None):
    item_x = []
    try:
        user_x = user_d[row.user_id]
    except KeyError:
        score =-999
    try:
        item_x.append(item_d[row.item_id])
        score = model.predict(user_x, item_x, item_features=item_f, num_threads=4)
        score = score[0]
    except KeyError:
        score = -999
    row['light_fm_score'] = score
    if(row.item_id == int(row.reference)):
        label = 1
    else:
        label = 0
    row['label'] = label
    return row

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
    #print('No prediction for #' + str(df_missed.shape[0]) + 'items')
    df_out_nation = df_test_nation[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]
    return df_out_nation

def fill_recs(imp):
    l = imp.split('|')
    return f.list_to_space_string(l)

def generate_user_features(df, user_dict):
    df['present'] = 1
    #print('# of duplicates: ' + str(starting_user - df_user_features.shape[0]))
    df_user_features = df[['user_id', 'platform', 'present']]
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

    return csr



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
    #x = sparse.csr_matrix(interactions.values)
    model = LightFM(no_components= params.ncomponents, loss=params.lossfunction, k=params.mfk, learning_schedule=params.learningschedule, learning_rate=params.learningrate, user_alpha=1e-6, item_alpha=1e-6)
    model.fit(interactions,epochs=params.epochs,num_threads = n_jobs, item_features=item_f, user_features=user_f)
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
    