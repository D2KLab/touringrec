import pandas as pd
import numpy as np
import functions as f
from tqdm import tqdm
import operator


def delete_dataset_error(df_filtered):
    print(str(df_filtered.shape[0]))
    df_filtered = df_filtered[df_filtered['reference'] != '8980878']
    df_filtered = df_filtered[df_filtered['reference'] != '6766746']
    df_filtered = df_filtered[df_filtered['reference'] != '10941440']
    df_filtered = df_filtered[df_filtered['reference'] != '4466176']
    df_filtered = df_filtered[df_filtered['reference'] != '6953208']

    return df_filtered

def clean_test_set(df):
    mask = (df["action_type"] == "clickout item") | (df["action_type"] == "interaction item rating") | (df["action_type"] == "interaction item image") | (df["action_type"] == "interaction item deals")
    df_cleaned = df[mask]
    df_test = df_cleaned.drop(df_cleaned[(df_cleaned['action_type'] == "clickout item") & (df_cleaned['reference'].isnull())].index)
    return df_test;

    

def get_rec_matrix(df_train, df_test, **kwargs):
    df_train = df_train.head(100000)
    df_train = delete_dataset_error(df_train)
    df_test = delete_dataset_error(df_test)
    file_metadata = kwargs.get('file_metadata', None)
    print('Reading metadata file' + str(file_metadata))
    df_metadata = pd.read_csv(file_metadata)
    print('Take relevant action of test set')
    print('#element test: ' + str(df_test.shape[0]))
    df_test_cleaned = clean_test_set(df_test)
    print('#element test cleaned: ' + str(df_test_cleaned.shape[0]))
    print('Joining train and cleaned test set')
    print('#element train before: ' + str(df_train.shape[0]))
    df_train = pd.concat([df_train, df_test_cleaned], ignore_index=True)
    print('#element train after: ' + str(df_train.shape[0]))
    print('Selecting only hotel referenced')
    mask = (df_train["action_type"] == "clickout item") | (df_train["action_type"] == "interaction item rating") | (df_train["action_type"] == "interaction item image") | (df_train["action_type"] == "interaction item deals")
    #mask = df["action_type"] == "clickout item"
    df_filtered = df_train[mask]
    df_filtered = pd.concat([df_filtered, df_test_cleaned])
    df_filtered['reference'] = df_filtered['reference'].astype(int)
    df_feature_matrix = generate_feature_matrix(df_metadata, df_filtered)
    df_preference_matrix = generate_preference_matrix(df_metadata, df_filtered, df_feature_matrix)
    print('Preference matrix:')
    print(df_preference_matrix.head())
    return

def generate_feature_matrix(df_metadata, df_filtered):

    #df_metadata_clean = df_metadata[df_metadata.item_id.isin(df_filtered.reference)]
    print('Exploding dataframe')
    df_exploded = f.explode(df_metadata, 'properties',flag_conversion=False)
    df_exploded['present'] = 1
    # Creating interaction matrix using rating data
    print('Creating the feature matrix')
    df_feature_matrix = f.create_interaction_matrix(df = df_exploded,
                                                    user_col = 'item_id',
                                                    item_col = 'properties',
                                                    rating_col = 'present')
    return df_feature_matrix.transpose()
    
def generate_preference_matrix(df_metadata, df_filtered, df_feature_matrix):
    properties = f.get_all_properties(df_metadata)
    print('Generating the empty dataframe for preferences')
    tot_user = df_filtered['user_id'].drop_duplicates().tolist()
    matrix_interaction = pd.DataFrame(index = properties, columns = tot_user).fillna(0)
    print('Start filling the matrix with user preferences')
    for index, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0]):
        my_hotel = row['reference']
        try:
            feature_array = df_feature_matrix[my_hotel]
            my_user = row['user_id']
            matrix_interaction[my_user] = matrix_interaction[my_user] + feature_array
        except KeyError as e:
            print(e)
    return matrix_interaction


def generate_submission(df_test, feature_matrix, preference_matrix):
    df_test = f.get_submission_target(df_test)
    print('Calculate submission')
    cols = ['user_id', 'session_id', 'timestamp', 'step', 'item_recommendations']
    df_out = pd.DataFrame(columns=cols)
    for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        my_user = row['user_id']
        my_session = row['session_id']
        my_timestamp = row['timestamp']
        my_step = row['step']
        my_impressions = row['impressions']

        # Explode impressions
        temp = my_impressions.split('|')
        impressions = []
        for i in temp:
            impressions.append(i)
        
        # Get the score for each impression item
        scores = []
        for i in impressions:
            if(my_user in preference_matrix.index): # Trovo le preferenze dell'utente
                user_preferences = preference_matrix[my_user]
            else:
                user_preferences = 0
            if(i in feature_matrix.columns): # Trovo le feature
                hotel_feature = feature_matrix[i]
            else: #Sarà da sostituire con il mostpopular o con qualche altro meccanismo
                hotel_feature = 0    
            score = user_preferences * hotel_feature
            scores.append(score)
        score_for_hotel = dict(zip(impressions, score))
        sorted_hotel = sorted(score_for_hotel.items(), key=operator.itemgetter(1))


