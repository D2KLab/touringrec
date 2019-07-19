import pandas as pd
import numpy as np

MERGE_COLS = ["user_id", "session_id", "timestamp", "step"]

def complete_ensemble(row):
    if(row.item_recommendations_rule == ''):
        return row.item_recommendations_imp
    else:
        return row.item_recommendations_rule

    
def get_clickouts(df_test):
    df_test['step_max'] = df_test.groupby(['user_id'])['step'].transform(max)
    df_clickout = df_test[((df_test['step_max']-1) == df_test['step']) & (df_test['action_type'] == 'clickout item')]
    del df_clickout['step_max']
    return df_clickout

def get_submission_target(df):
    """Identify target rows with missing click outs."""

    mask = df["reference"].isnull() & (df["action_type"] == "clickout item")
    df_out = df[mask]

    return df_out


#df_mf = pd.read_csv('submission_mf_xgboost_new_validation.csv')
df_rule = pd.read_csv('rule_based_FR_reconstruct.csv')
df_test = pd.read_csv('test_off.csv')
df_rule = df_rule.loc[:, ~df_rule.columns.str.contains('^Unnamed')]
df_clickout = get_clickouts(df_test)
df_rule = df_rule[~df_rule['session_id'].isin(df_clickout['session_id'])]
print(df_rule.head())
df_test = get_submission_target(df_test)
df_test = df_test[MERGE_COLS + ['impressions']]
df_test = df_test.rename(columns={'impressions':'item_recommendations'})
df_test['item_recommendations'] = df_test['item_recommendations'].apply(lambda x : " ".join(list(x.split('|'))))
print(df_test.head())
df_merged = (
    df_test
    .merge(df_rule,suffixes=('_imp', '_rule'),
           left_on=MERGE_COLS,
           right_on=MERGE_COLS,
           how="left")
    )

df_merged = df_merged.fillna('')
print(str(df_merged[df_merged['item_recommendations_rule'] == ''].shape[0]))
print(df_merged.head(100))
df_merged['item_recommendations'] = df_merged.apply(lambda x : complete_ensemble(x), axis = 1)
print(df_merged.head())
del df_merged['item_recommendations_rule']
del df_merged['item_recommendations_imp']
print(df_merged.head())

df_merged.to_csv('sub_ensamble_logic_rule.csv')