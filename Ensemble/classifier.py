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



def get_session_stats(df_gt):
    '''
        Input -> ground truth dataframe
        Output -> df: session_id|session_length|n_hotel|sparsity
    '''
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
test = [[3,1]]
label = svmModel.predict_proba(test)
label