import pandas as pd
import numpy as np
import functions as f
from tqdm import tqdm
import operator
import collections as cl




def delete_dataset_error(df_filtered):
    print(str(df_filtered.shape[0]))
    df_filtered = df_filtered[df_filtered['reference'] != '8980878']
    df_filtered = df_filtered[df_filtered['reference'] != '6766746']
    df_filtered = df_filtered[df_filtered['reference'] != '10941440']
    df_filtered = df_filtered[df_filtered['reference'] != '4466176']
    df_filtered = df_filtered[df_filtered['reference'] != '6953208']

    return df_filtered

def clean_test_set(df):
    mask = (df["action_type"] == "clickout item") | (df["action_type"] == "interaction item rating") | (df["action_type"] == "search for item")|(df["action_type"] == "interaction item image") | (df["action_type"] == "interaction item deals")
    df_cleaned = df[mask]
    df_test = df_cleaned.drop(df_cleaned[(df_cleaned['action_type'] == "clickout item") & (df_cleaned['reference'].isnull())].index)
    return df_test;

    

def get_rec_matrix(df_train, df_test, **kwargs):
    subm_csv = "submission_popular.csv"
    df_train = delete_dataset_error(df_train)
    df_train = df_train.head(10)
    df_test = delete_dataset_error(df_test)
    file_metadata = kwargs.get('file_metadata', None)
    print('Reading metadata file' + str(file_metadata))
    df_metadata = pd.read_csv(file_metadata)
    #df_metadata = df_metadata.head(10)
    print('Take relevant action of test set')
    print('#element test: ' + str(df_test.shape[0]))
    df_test_cleaned = clean_test_set(df_test)
    #df_test_cleaned = df_test_cleaned.head(1000)
    print('#element test cleaned: ' + str(df_test_cleaned.shape[0]))
    print('Joining train and cleaned test set')
    print('#element train before: ' + str(df_train.shape[0]))
    df_train = pd.concat([df_train, df_test_cleaned], ignore_index=True)
    print('#element train after: ' + str(df_train.shape[0]))
    print('Selecting only hotel referenced')
    mask = (df_train["action_type"] == "clickout item") | (df_train["action_type"] == "interaction item rating") | (df_train["action_type"] == "search for item") | (df_train["action_type"] == "interaction item image") | (df_train["action_type"] == "interaction item deals")
    #mask = df["action_type"] == "clickout item"
    df_filtered = df_train[mask]
    df_filtered = df_filtered.head(10)
    df_filtered = pd.concat([df_filtered, df_test_cleaned])
    df_filtered['reference'] = df_filtered['reference'].astype(int)
    #df_feature_matrix = generate_feature_matrix(df_metadata)
    df_feature_dict = f.encode_features(df_metadata)
    #df_feature_matrix.to_csv('feature_matrix.csv')
    #df_feature_matrix = pd.read_csv('feature_matrix.csv')
    df_preference_matrix = generate_preference_matrix_dict(df_metadata, df_filtered, df_feature_dict)
    print('Preference matrix:')
    print(df_preference_matrix.head())
    df_out = generate_submission(df_test, df_feature_dict, df_preference_matrix)
    print('First element of the submission:')
    print(df_out.head())
    print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)
    return

def generate_preference_matrix_dict(df_metadata, df_filtered, feature_dict):
    properties = f.get_all_properties(df_metadata)
    print('Generating the empty dataframe for preferences')
    tot_user = df_filtered['user_id'].drop_duplicates().tolist()
    matrix_interaction = pd.DataFrame(index = properties, columns = tot_user).fillna(0)
    print('Start filling the matrix with user preferences')
    for index, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0]):
        my_hotel = row['reference']
        if(int(my_hotel) in feature_dict): # Trovo le feature
            feature_array = feature_dict.get(my_hotel)
            my_user = row['user_id']
            matrix_interaction[my_user] = matrix_interaction[my_user] + feature_array
    return matrix_interaction

def generate_feature_matrix(df_metadata):

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

def list_to_space_string(l):
    s = " ".join(l)
    return s

# Funzione utilizzata nel caso di matrice di feature

# def generate_submission(df_test, feature_matrix, preference_matrix):
#     df_test = f.get_submission_target(df_test)
#     print('Calculate submission')
#     cols = ['user_id', 'session_id', 'timestamp', 'step', 'item_recommendations']
#     df_out = pd.DataFrame(columns=cols)
#     for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
#         my_user = row['user_id']
#         my_session = row['session_id']
#         my_timestamp = row['timestamp']
#         my_step = row['step']
#         my_impressions = row['impressions']

#         # Explode impressions
#         impressions = my_impressions.split('|')

        
#         # Get the score for each impression item
#         scores = []
#         for i in impressions:
#             if(my_user in preference_matrix.columns): # Trovo le preferenze dell'utente
#                 user_preferences = preference_matrix[my_user].values
#                 #print(type(user_preferences))
#             else:
#                 not_present = True
#             if(int(i) in feature_matrix.columns): # Trovo le feature
#                 hotel_feature = feature_matrix[int(i)].values
#                 #print(type(hotel_feature))
#             else: #Sarà da sostituire con il mostpopular o con qualche altro meccanismo
#                 not_present = True            
#             if(not_present == True):
#                 score = 0
#             else:
#                 score = np.dot(user_preferences, hotel_feature.transpose())
#             scores.append(score)
#             not_present = False
#         score_for_hotel = dict(zip(impressions, scores))
#         sorted_x = sorted(score_for_hotel.items(), key=operator.itemgetter(1))
#         sorted_dict = cl.OrderedDict(sorted_x)
#         hotel_rec = list_to_space_string(list(sorted_dict.keys()))
#         #print(hotel_rec)
#         df_out = df_out.append({'user_id': my_user, 'session_id': my_session, 'timestamp': my_timestamp, 'step': my_step, 'item_recommendations': hotel_rec}, ignore_index=True)
    
#     return df_out

def generate_submission(df_test, feature_dict, preference_matrix):
    df_popular_nation = pd.read_csv('submission_popular_nation.csv')
    feature_used = 0
    not_present = False
    df_test = f.get_submission_target(df_test)
    #df_test = df_test.head(1000)
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
        impressions = my_impressions.split('|')
        # Get the score for each impression item
        scores = []
        for i in impressions:
            if(my_user in preference_matrix.columns): # Trovo le preferenze dell'utente
                user_preferences = preference_matrix[my_user].values
            else:
                not_present = True
            if(int(i) in feature_dict): # Trovo le feature
                hotel_feature = feature_dict.get(int(i))
            else: #Sarà da sostituire con il mostpopular o con qualche altro meccanismo
                not_present = True  
            if(not_present == True):
                # Assegno uno score reciproco 1/posizione nel most popular
                temp = df_popular_nation[(df_popular_nation['user_id'] == my_user) & (df_popular_nation['session_id'] == my_session)]
                temp_impressions = temp['item_recommendations']
                temp_impressions_array = temp_impressions.values[0].split(" ")
                for j in range(0, len(temp_impressions_array)):
                    if(temp_impressions_array[j] == i):
                        score = (1 / (j+1)) * 2 #Parameters to be tuned
                        break
            else:
                feature_used = feature_used + 1
                score = np.dot(user_preferences, hotel_feature.transpose())
            scores.append(score)
            not_present = False
        score_for_hotel = dict(zip(impressions, scores))
        sorted_x = sorted(score_for_hotel.items(), key=operator.itemgetter(1))
        sorted_dict = cl.OrderedDict(sorted_x)
        hotel_rec = list_to_space_string(list(sorted_dict.keys()))
        #print(hotel_rec)
        df_out = df_out.append({'user_id': my_user, 'session_id': my_session, 'timestamp': my_timestamp, 'step': my_step, 'item_recommendations': hotel_rec}, ignore_index=True)
    print('The hotel feature have been used #:' + str(feature_used))
    return df_out


