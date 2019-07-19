import numpy as np
import pandas as pd
import argparse

# Scores for rule-based algo computed when analysing the tr data
P_list = [0.485, 0.441, 0.342, 0.327, 0.324, 0.27]
P_pr_sess = 0.2
## Loading the data
def load_data(path):
    df_test_dev_split = pd.read_csv(path + 'test_dev.csv')
    return df_test_dev_split

#### Data Preprocessing
def preprocess_data(df_test_dev_split):
    ## add nan to df_test_dev_split
    df_test_dev_split.reference.fillna('NAN', inplace=True)
    df_test_dev_split_eval = df_test_dev_split[df_test_dev_split['action_type'] == 'clickout item']
    ##########################
    ## get matrix of test data
    matrix_test_dev_eval = df_test_dev_split_eval[['user_id', 'session_id', 'step', 'timestamp',
                                                'impressions', 'reference', 'prices', 'current_filters']].values
    ##########################
    User_Session_test_dev = list()
    for el in matrix_test_dev_eval:
        User_Session_test_dev.append(str(el[0])+str(el[1])+str(el[2]))
    ###########################
    matrix_test_dev = df_test_dev_split[['user_id', 'session_id', 'step', 'impressions', 'prices',
                'action_type', 'reference', 'platform', 'city', 'device', 'current_filters']].values
    ###########################
    ## Creating dictionnary in order to retrieve past actions of same session
    dict_test = dict()
    for el in matrix_test_dev:
        dict_test[str(el[0])+str(el[1])+str(el[2])] = str(el[3]) + ';' + str(el[4]) + ';' + str(el[5]) + 
        ';' + str(el[6]) + ';' + str(el[7]) + ';' + str(el[8]) + ';' + str(el[9]) + ';' + str(el[10])
    return dict_test, User_Session_test_dev, matrix_test_dev_eval
    
def rule_based_algo(matrix_test_dev_eval, User_Session_test_dev): 
    ###########################
    # Rule-based algorithm
    list_impressions = list();list_user_id = list();list_session_id = list();list_hotel_id = list();list_timestamp = list()
    list_step = list();list_score = list()
    ## Loop Over all the submissions
    matrix_eval = matrix_test_dev_eval
    User_Session_Step_eval = User_Session_test_dev
    for i in range(len(matrix_eval)):
        if not(i%50000):
            print(i)
        # current step information
        user_id = matrix_eval[i, 0]
        session_id = matrix_eval[i, 1]
        timestamp = matrix_eval[i, 3]
        step = matrix_eval[i, 2]
        # impression list
        impressions_test = matrix_eval[i,4].split('|')
        current_step = matrix_eval[i,2]
        # Retrieve the action in the main test df
        EL = dict_test[User_Session_Step_eval[i]].split(';')    
        if user_id in dict_click_user: 
            clickouts = list(set(dict_click_user[user_id]))
        if current_step < 2:
            for j, hotel_id in enumerate(impressions_test):
                list_user_id.append(user_id)
                list_session_id.append(session_id)
                list_hotel_id.append(hotel_id)
                list_timestamp.append(timestamp)
                list_step.append(step)
                if hotel_id in clickouts:
                    list_score.append(P_pr_sess)
                else:
                    list_score.append(0)
        # if not we see the previous actions
        else:
            step_moins_1 = current_step - 1
            user_session = User_Session_Step_eval[i][0:len(User_Session_Step_eval[i])-len(str(current_step))]
            EL_Moins_1 = dict_test[str(user_session) + str(step_moins_1)].split(';')
            action_type_Moins_1 = EL_Moins_1[2]   
            case_poi += 1 
            if EL_Moins_1[3] in impressions_test:
                for j, hotel_id in enumerate(impressions_test):
                    list_user_id.append(user_id)
                    list_session_id.append(session_id)
                    list_hotel_id.append(hotel_id)
                    list_timestamp.append(timestamp)
                    list_step.append(step)
                    if hotel_id == EL_Moins_1[3]:
                        list_score.append(P_list[0])
                    elif hotel_id in clickouts:
                        list_score.append(P_pr_sess)
                    else:
                        list_score.append(0)
            elif step_moins_1 <2:
                for j, hotel_id in enumerate(impressions_test):
                    list_user_id.append(user_id)
                    list_session_id.append(session_id)
                    list_hotel_id.append(hotel_id)
                    list_timestamp.append(timestamp)
                    list_step.append(step)
                    if hotel_id in clickouts:
                        list_score.append(P_pr_sess)
                    else:
                        list_score.append(0)            
            else:
                step_moins_2 = step_moins_1 - 1
                user_session = User_Session_Step_eval[i][0:len(User_Session_Step_eval[i])-len(str(current_step))]
                EL_Moins_2 = dict_test[str(user_session) + str(step_moins_2)].split(';')
                if EL_Moins_2[3] in impressions_test:            
                    for j, hotel_id in enumerate(impressions_test):
                        list_user_id.append(user_id)
                        list_session_id.append(session_id)
                        list_hotel_id.append(hotel_id)
                        list_timestamp.append(timestamp)
                        list_step.append(step)
                        if hotel_id == EL_Moins_2[3]:
                            list_score.append(P_list[1])
                        elif hotel_id in clickouts:
                            list_score.append(P_pr_sess)
                        else:
                            list_score.append(0)
                elif step_moins_2 <2:
                    for j, hotel_id in enumerate(impressions_test):
                        list_user_id.append(user_id)
                        list_session_id.append(session_id)
                        list_hotel_id.append(hotel_id)
                        list_timestamp.append(timestamp)
                        list_step.append(step)
                        if hotel_id in clickouts:
                            list_score.append(P_pr_sess)
                        else:
                            list_score.append(0)                  
                else:
                    step_moins_3 = step_moins_2 - 1
                    user_session = User_Session_Step_eval[i][0:len(User_Session_Step_eval[i])-len(str(current_step))]
                    EL_Moins_3 = dict_test[str(user_session) + str(step_moins_3)].split(';')
                    if EL_Moins_3[3] in impressions_test: 
                        for j, hotel_id in enumerate(impressions_test):
                            list_user_id.append(user_id)
                            list_session_id.append(session_id)
                            list_hotel_id.append(hotel_id)
                            list_timestamp.append(timestamp)
                            list_step.append(step)
                            if hotel_id == EL_Moins_3[3]:
                                list_score.append(P_list[2])
                            elif hotel_id in clickouts:
                                list_score.append(P_pr_sess)
                            else:
                                list_score.append(0)
                    else:
                        for j, hotel_id in enumerate(impressions_test):
                            list_user_id.append(user_id)
                            list_session_id.append(session_id)
                            list_hotel_id.append(hotel_id)
                            list_timestamp.append(timestamp)
                            list_step.append(step)
                            if hotel_id in clickouts:
                                list_score.append(P_pr_sess)
                            else:
                                list_score.append(0)   
    ### Output result of Rule-based algo

    df_data = {'user_id': list_user_id,
            'session_id': list_session_id,
            'hotel_id': list_hotel_id,
            'timestamp': list_timestamp,
            'step': list_step,
            'score': list_score}

    return df_data

def main():

    # Rule-based arguments
    parser = argparse.ArgumentParser(description='rule-based')  
    parser.add_argument('--path_to_read', type=str, default='../data/')
    parser.add_argument('--path_to_save', type=str, default='../data/')

    args = parser.parse_args()
    print(args)
    
    # Load the data
    df_test_dev_split = load_data(args.path_to_read)

    # Preprocess the data for the rule-based algo
    dict_test, User_Session_test_dev, matrix_test_dev_eval = preprocess_data(df_test_dev_split)

    # Run rule-based algorithm
    df_data = rule_based_algo(matrix_test_dev_eval, User_Session_test_dev)

    ## Save rule-based scores
    df_to_submit = pd.DataFrame.from_dict(df_data)
    df_to_submit.to_csv(args.path_to_save + 'Rule_based_Test_Dev.csv')

if __name__ == '__main__':
    main()    