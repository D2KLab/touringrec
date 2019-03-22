import pandas as pd
import numpy as np
import functions as f

df_train = pd.read_csv('data/prova.csv')

session_train, session_test = f.split_group_random(df_train, 0.8, 'session_id')

f.split_group_csv(df_train, session_train, session_test, 'session_id')
