import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import classification_report
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score

MERGE_COLS = ["user_id", "session_id", "timestamp", "step"]

def generate_rranks_range(start, end):
    """Generate reciprocal ranks for a given list length."""

    return 1.0 / (np.arange(start, end) + 1)

def read_into_df(file):
    """Read csv file into data frame."""
    df = (
        pd.read_csv(file)
            .set_index(['user_id', 'session_id', 'timestamp', 'step'])
    )

    return df

def convert_string_to_list(df, col, new_col):
    """Convert column from string to list format."""
    fxn = lambda arr_string: [int(item) for item in str(arr_string).split(" ")]

    mask = ~(df[col].isnull())

    df[new_col] = df[col]
    df.loc[mask, new_col] = df[mask][col].map(fxn)

    return df

def score_submissions(subm_csv, gt_csv, objective_function):
    """Score submissions with given objective function."""

    print(f"Reading ground truth data {gt_csv} ...")
    df_gt = read_into_df(gt_csv)

    print(f"Reading submission data {subm_csv} ...")
    df_subm = read_into_df(subm_csv)
    print('Submissions')
    print(df_subm.head(10))
    # create dataframe containing the ground truth to target rows
    cols = ['reference', 'impressions', 'prices']
    df_key = df_gt.loc[:, cols]

    # append key to submission file
    df_subm_with_key = df_key.join(df_subm, how='inner')
    print(df_subm_with_key.head())
    df_subm_with_key.reference = df_subm_with_key.reference.astype(int)
    df_subm_with_key = convert_string_to_list(
        df_subm_with_key, 'item_recommendations', 'item_recommendations'
    )

    # score each row
    df_subm_with_key['score'] = df_subm_with_key.apply(objective_function, axis=1)
    df_subm_with_key.to_csv('borda.csv')
    print(df_subm_with_key)
    mrr = df_subm_with_key.score.mean()

    return mrr
    
def get_reciprocal_ranks(ps):
    """Calculate reciprocal ranks for recommendations."""
    mask = ps.reference == np.array(ps.item_recommendations)

    if mask.sum() == 1:
        rranks = generate_rranks_range(0, len(ps.item_recommendations))
        return np.array(rranks)[mask].min()
    else:
        return 0.0

def remove_null_clickout(df):
    """
    Remove all the occurences where the clickout reference is set to null (Item to predict)
    """
    df = df.drop(df[(df['action_type'] == "clickout item") & (df['reference'].isnull())].index)
    return df

def get_session_stats(df_gt):
    '''
        Input -> ground truth dataframe
        Output -> df: session_id|session_length|n_hotel|sparsity
    '''
    df_gt = remove_null_clickout(df_gt)
    mask = (df_gt["action_type"] == "clickout item") | (df_gt["action_type"] == "interaction item rating") | (df_gt["action_type"] == "search for item")|(df_gt["action_type"] == "interaction item image") | (df_gt["action_type"] == "interaction item deals")
    df_gt = df_gt[mask] 
    s = df_gt.groupby('session_id').agg({
        'session_id': "count",
        'reference': 'nunique'
    })
    s.columns = ['session_length', 'n_hotel']
    s['sparsity'] = s.apply(lambda x: x.n_hotel / x.session_length, axis=1)
    return s

def calculate_best(rank_mf, rank_rnn):
    if(rank_mf > rank_rnn):
        return 0
    else:
        return 1

def splitTrainValidationTest(samples, labels, nTrain, nValidation, nTest):
    fracTrain = nTrain / 10  # percentage of training data (e.g. 5:2:3 -> 0.5 of train)
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=fracTrain, random_state=1)
    fracValidation = nValidation / (10 - nTrain)  # Take the percentage of validation
    x_test, x_validation, y_test, y_validation = train_test_split(samples, labels, test_size=fracValidation,
                                                                  random_state=1)
    return x_train, x_test, x_validation, y_train, y_test, y_validation

'''
    This function receive the training set and the test set and will train the chosen model with the C parameters provided
'''
def evaluateModel(x_train, y_train, myC, x_test, y_test, kernel_type):

    svc = svm.SVC(kernel=kernel_type, C=myC)
    svc.fit(x_train, y_train)
    # Evaluate accuracy score
    pr = svc.predict(x_test)
    score = accuracy_score(y_test, pr)
    print('SVM with ' + kernel_type + " kernel accuracy: " + str(score))
    return

def testClassifier(kerneltype, bestC, bestGamma, x_train, y_train, x_test, y_test):
    svc = svm.SVC(kernel=kerneltype, random_state=0, C=bestC, gamma=bestGamma, probability=True)
    svc.fit(x_train, y_train)
    #plot_decision_regions(x_train, y_train, clf=svc, legend=2)
    #plt.show()
    pr = svc.predict(x_test)
    score = accuracy_score(y_test, pr)
    return score, svc
'''
'''
def kFoldParamEvaluation(x_train, y_train, x_test, y_test, c_values, gamma_values):
    tuned_params = [{'kernel': ['rbf','linear'], 'gamma': gamma_values, 'C': c_values }]
    clf = GridSearchCV(svm.SVC(), tuned_params, cv=5)
    clf.fit(x_train, y_train)
    print("Best gamma and C found: ")
    print()
    print(clf.best_params_)
    y_true, y_predict = y_test, clf.predict(x_test)
    #print(classification_report(y_true, y_predict))
    return clf.best_params_

def generate_labels(df):
    df['label'] = df.apply(lambda x: calculate_best(x.rank_mf, x.rank_rnn), axis = 1)
    return df

def get_ensemble(row, svm_model):
    session_length = row.session_length
    sparsity  = row.sparsity
    if(row.item_recommendations_mf == ''):
        return row.item_recommendations_rnn
    if(row.item_recommendations_rnn == ''):
        return row.item_recommendations_mf
    mf_rec = row.item_recommendations_mf.split(' ')
    rnn_rec = row.item_recommendations_rnn.split(' ')
    test = [[int(session_length), float(sparsity)]]
    confidence = svm_model.predict_proba(test)
    result_list = []    
    
    if confidence[0][0] > 0.5: #MF migliore
        result_list.append(mf_rec[0])
        rnn_rec.remove(mf_rec[0])
        result_list = result_list + rnn_rec
    else: #RNN migliore
        result_list = rnn_rec
    return ' '.join(result_list)

df_gt = pd.read_csv('gt.csv')
df_label = pd.read_csv('upperbound.csv')
session_stats = get_session_stats(df_gt)
df_merged = (
    df_label
    .merge(session_stats,suffixes=('_mf', '_rnn'),
           left_on='session_id',
           right_on='session_id',
           how="left")
    )
print('Remove ties')
df_merged = df_merged[df_merged['rank_mf'] != df_merged['rank_rnn']]
print('Remove single click')
df_merged = df_merged[df_merged['session_length'] != 1]
df_data_svm = generate_labels(df_merged)
print(df_data_svm[['session_id', 'session_length', 'sparsity']].head())
x = df_data_svm[['session_length', 'sparsity']].values
y = df_data_svm[['label']].values.ravel()
print('INPUT')
print(x.shape)
print('LABEL')
print(y.shape)

# Find the best parameters of the RBF kernel

myGamma = [0.01, 0.1, 1, 10]
myC = [0.01, 0.1, 1, 10]
x_train, x_test, x_validation, y_train, y_test, y_validation = splitTrainValidationTest(x, y, 7, 1, 2)
best_params = kFoldParamEvaluation(x_train, y_train, x_validation, y_validation, myC, myGamma)
bestScore, svmModel = testClassifier(best_params['kernel'], best_params["C"], best_params["gamma"], x_train, y_train, x_test, y_test)
print(bestScore)
#test = [[3,1]]
#label = svmModel.predict_proba(test)
df_test = pd.read_csv('test_1.csv')
df_mf = pd.read_csv('submission_matrixfactorization_1.csv')
df_rnn = pd.read_csv('submission_rnn_1.csv')
df_rnn = df_rnn.loc[:, ~df_rnn.columns.str.contains('^Unnamed')]
df_data = get_session_stats(df_test)
df_merged = (
    df_mf
    .merge(df_data,
           left_on='session_id',
           right_on='session_id',
           how="left")
    )
df_merged = (
    df_merged
    .merge(df_rnn,suffixes=('_mf', '_rnn'),
           left_on=MERGE_COLS,
           right_on=MERGE_COLS,
           how="left")
    )
df_merged = df_merged.fillna('')
df_merged['item_recommendations'] = df_merged.apply(lambda x : get_ensemble(x, svmModel), axis=1)
submission_file = 'submission_ensemble.csv'
gt_file = 'gt.csv'
df_merged.to_csv(submission_file)
mrr =score_submissions(submission_file, gt_file, get_reciprocal_ranks)
#print(df_merged.head())
print('Score: ' + str(mrr))