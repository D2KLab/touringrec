import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f

GR_COLS = ["user_id", "session_id", "timestamp", "step"]


def get_popularity(df):
    """Get number of clicks that each item received in the df."""

    mask = df["action_type"] == "clickout item"
    #mask = (df["action_type"] == "clickout item") | (df["action_type"] == "interaction item rating") | (df["action_type"] == "interaction item info") | (df["action_type"] == "interaction item image") | (df["action_type"] == "interaction item deals")
    #mask = (df["action_type"] == "clickout item") | (df["action_type"] == "interaction item rating") | (df["action_type"] == "interaction item image") | (df["action_type"] == "interaction item deals")
    df_clicks = df[mask]
    df_item_clicks = (
        df_clicks
        .groupby("reference")
        .size()
        .reset_index(name="n_clicks")
        .transform(lambda x: x.astype(int))
    )

    return df_item_clicks

def get_popularity_by_nation(df, df_expl, df_popular, w_nation, w_all):
    # Generate most popular per nation table
    mask = df["action_type"] == "clickout item"
    #mask = (df["action_type"] == "clickout item") | (df["action_type"] == "interaction item rating") | (df["action_type"] == "interaction item image") | (df["action_type"] == "interaction item deals")
    df_clicks = df[mask]
    df_item_clicks = (
    df_clicks
        .groupby(["reference", "platform"])
        .size()
        .reset_index(name="n_clicks_nation")
    )
    df_item_clicks['n_clicks_nation'] = df_item_clicks['n_clicks_nation'].astype(int)
    df_item_clicks['reference'] = df_item_clicks['reference'].astype(int)
    print("First elements of click per nation")
    print(df_item_clicks.head())
    # Join most popular per nation and exploded 
    df_expl_clicks_nations = (
    df_expl[GR_COLS + ["impressions"] + ["platform"]]
    .merge(df_item_clicks,
           left_on=["impressions", "platform"],
           right_on=["reference", "platform"],
           how="left")
    )
    df_expl_clicks_nations = df_expl_clicks_nations.fillna(0)
    print("First elements of click per nation joined with exploded df")
    print(df_expl_clicks_nations.head())
    #Sum of the two colums tot_click + click_nation
    df_click_weighted = (
    df_expl_clicks_nations[GR_COLS + ["impressions"] + ["n_clicks_nation"] + ["platform"]]
    .merge(df_popular,
           left_on="impressions",
           right_on="reference",
           how="left")
    )

    df_click_weighted = df_click_weighted.fillna(0)
    df_click_weighted['n_clicks'] = w_nation * df_click_weighted['n_clicks_nation'] + w_all * df_click_weighted['n_clicks']
    del df_click_weighted['n_clicks_nation']
    print("First elements of click per nation joined with most popular")
    print(df_click_weighted.head())
    return df_click_weighted


def group_concat(df, gr_cols, col_concat):
    """Concatenate multiple rows into one."""

    df_out = (
        df
        .groupby(gr_cols)[col_concat]
        .apply(lambda x: ' '.join(x))
        .to_frame()
        .reset_index()
    )

    return df_out

def calc_recommendation_nation(df_pop):

    print("Popular input")
    print(df_pop.head())
    df_out = (
        df_pop
        .assign(impressions=lambda x: x["impressions"].apply(str))
        .sort_values(GR_COLS + ["n_clicks"],
                     ascending=[True, True, True, True, False])
    )

    df_out = group_concat(df_out, GR_COLS, "impressions")
    df_out.rename(columns={'impressions': 'item_recommendations'}, inplace=True)
    print(df_out)
    return df_out

def calc_recommendation(df_expl, df_pop):
    """Calculate recommendations based on popularity of items.

    The final data frame will have an impression list sorted according to the number of clicks per item in a reference data frame.

    :param df_expl: Data frame with exploded impression list
    :param df_pop: Data frame with items and number of clicks
    :return: Data frame with sorted impression list according to popularity in df_pop
    """
    print("Exploded input")
    print(df_expl.head())
    print("Popular input")
    print(df_pop.head())
    df_expl_clicks = (
        df_expl[GR_COLS + ["impressions"]]
        .merge(df_pop,
               left_on="impressions",
               right_on="reference",
               how="left")
    )
    
    print("DEBUG: campi che ci servono")
    print(df_expl_clicks.head())
    df_out = (
        df_expl_clicks
        .assign(impressions=lambda x: x["impressions"].apply(str))
        .sort_values(GR_COLS + ["n_clicks"],
                     ascending=[True, True, True, True, False])
    )

    df_out = group_concat(df_out, GR_COLS, "impressions")
    df_out.rename(columns={'impressions': 'item_recommendations'}, inplace=True)
    print(df_out.head())
    return df_out
    

def get_rec_nation(df_train, df_test, **kwargs):

    w_nation = kwargs.get('w_nation', 1)
    w_base = kwargs.get('w_base', 0.01)
    f.send_telegram_message("Starting base solution by nation with w_nation = " + str(w_nation) + " and w_base: " + str(w_base))

    subm_csv = "submission_basesolution_nation.csv"
    df_test_cleaned = f.remove_null_clickout(df_test)
    df_train = pd.concat([df_train, df_test_cleaned], ignore_index=True, sort=False)
    print("Get popular items...")
    df_popular = get_popularity(df_train)
    print("Dataset of popular starts with:")
    print(df_popular.head())

    print("Identify target rows...")
    df_target = f.get_submission_target(df_test)
    print("Dataset of target starts with:")
    print(df_target.head())

    print("Get recommendations...")
    df_expl = f.explode(df_target, "impressions")
    df_popular_nation = get_popularity_by_nation(df_train, df_expl, df_popular, w_nation, w_base)
    df_out = calc_recommendation_nation(df_popular_nation)

    #print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)

    print("Finished calculating recommendations.")

    return df_out

def get_rec_base(df_train, df_test, **kwargs):

    subm_csv = "submission_basesolution.csv"
    print("Get popular items...")
    df_popular = get_popularity(df_train)
    print("Dataset of popular starts with:")
    print(df_popular.head())

    print("Identify target rows...")
    df_target = f.get_submission_target(df_test)
    print("Dataset of target starts with:")
    print(df_target.head())

    print("Get recommendations...")
    df_expl = f.explode(df_target, "impressions")
    df_out = calc_recommendation(df_expl, df_popular)

    #print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)

    print("Finished calculating recommendations.")

    return df_out