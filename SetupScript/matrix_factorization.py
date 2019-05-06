import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f

def get_rec_matrix(df_train, df_test, **kwargs):
    
    # Select only a portion of dataset for testing purpose
    df_train = f.get_interaction_actions(df_train)
    df_test = f.get_interaction_actions(df_test, True)
    df_train = f.get_df_percentage(df_train, 0.05)
    print('Taken #: ' + str(df_train.shape[0]) + 'rows')
    df_test = f.get_df_percentage(df_test, 0.1)
    df_interactions = get_n_interaction(df_train)
    interaction_matrix = f.create_interaction_matrix(df_interactions,'user_id', 'reference', 'n_interactions')
    print(interaction_matrix.head())


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