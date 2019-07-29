import pandas as pd

def get_len(x):
    if x['reference'] == 'clickout action': 
        x
    return x

def get_avg_bef_ckout(df):
    df.groupby('session_id').apply(lambda x: get_len)

    return avg

def get_perc_single_act(df):
    df_count = df.groupby('session_id').size().reset_index(name = 'session_len')
    #print(df_count.head())

    df_filtered = df_count[df_count['session_len'] == 1]
    #print(df_filtered.head())

    n_sessions = len(df_count.index)
    print('Number of sessions is: ' + str(n_sessions))

    n_single_act = len(df_filtered.index)
    print('Number of single action sessions is: ' + str(n_single_act))

    somma = df_count['session_len'].sum()
    print(somma)
    print(somma / n_sessions)

    return n_single_act / n_sessions * 100

df_test_dev = pd.read_csv('./test_dev.csv')
df_test_off = pd.read_csv('./test_off.csv')

print('Dev stats:')
print(get_perc_single_act(df_test_dev))

print('Off stats:')
print(get_perc_single_act(df_test_off))
