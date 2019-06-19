import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import base_solution as bs
import matrix_factorization_metadata as mf_metadata
import matrix_factorization as mf
import mf_xgboost as mf_xg
import sys as sys
import argparse
import csv
import os.path
import json
from MFParameters import MF_Parameters

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', action="store", type=str, help="Choose the algorithm that you want to use", default='matrixfactorization')
parser.add_argument('--train', action="store", type=str, help="--train train.csv", default='train.csv')
parser.add_argument('--test', action="store", type=str, help="--test test.csv", default='test.csv')
parser.add_argument('--gt', action="store", type=str, help="--gt train.csv", default='gt.csv')
parser.add_argument('--metadata', action="store", type=str, help="Define the metadata file", default='hotel_prices.csv')
parser.add_argument('--localscore', action="store", type=int, help="0 -> Official score, 1 -> Local score", default=1)
parser.add_argument('--epochs', action="store", type=int, help="Define the number of epochs", default=30)
parser.add_argument('--ncomponents', action='store', type=int, help='MF: number of components', default=100)
parser.add_argument('--lossfunction', action='store', type=str, help='MF: define the loss function', default='warp-kos')
parser.add_argument('--learningrate', action='store', type=float, help='Choose the MF learning rate', default=0.5)
parser.add_argument('--learningschedule', action='store', type=str, help='MF: choose the learning schedule between adagrad and adadelta', default='adadelta')
parser.add_argument('--mfk', action='store', type=int, help='MF: parameter K', default=200)
parser.add_argument('--useralpha', action='store', type=float, help='User L2 penalty', default=0)
parser.add_argument('--itemalpha', action='store', type=float, help='Item L2 penalty', default=0)
parser.add_argument('--rho', action='store', type=float, help='Rho', default=0.95)
parser.add_argument('--epsilon', action='store', type=float, help='Epsilon', default=1e-06)
parser.add_argument('--maxsampled', action='store', type=float, help='Max Element sampled in warp loss', default=10)
parser.add_argument('--actions', nargs='+')
parser.add_argument('-v', '--verbose', action='store_true')

# Get all the parameters
args = parser.parse_args()
algorithm = args.algorithm
train = args.train
test = args.test
gt = args.gt
metadata = args.metadata
localscore = args.localscore
epochs = args.epochs
ncomponents = args.ncomponents
lossfunction = args.lossfunction
mfk = args.mfk
actions = args.actions
learningrate = args.learningrate
learningschedule = args.learningschedule
useralpha = args.useralpha
itemalpha = args.itemalpha
rho = args.rho
epsilon = args.epsilon
maxsampled = args.maxsampled
verbose = args.verbose
verboseprint = print if verbose else lambda *a, **k: None
# Convert actions in a correct list format
list_actions = []
actions_weights = {}
if actions != None:
    for i in actions:
        spl = i.split(' ')
        value = spl[-1]
        spl.remove(value)
        action = ' '.join(spl)
        actions_weights[action] = float(value)
        list_actions.append(' '.join(spl))
else:
    actions_weights = None
    list_actions = None
# Defining all the solutions implemented
solutions = {
    "basesolution": bs.get_rec_base,
    "basesolution_nation": bs.get_rec_nation,
    "matrixfactorization_metadata": mf_metadata.get_rec_matrix,
    "matrixfactorization": mf.get_rec_matrix,
    "mf_xgboost": mf_xg.get_rec_matrix,
}

params = MF_Parameters(epochs, ncomponents, lossfunction, mfk, learningrate, learningschedule, useralpha, itemalpha, rho, epsilon, maxsampled, actions_weights, list_actions)
params.print_params()
#importing and defining Datasets

#Train
verboseprint("Reading train set " + train)
df_train = pd.read_csv(train)
#Test
verboseprint("Reading test set " + test)
df_test = pd.read_csv(test)

verboseprint("Groud truth is: " + gt)
if metadata != None:
    verboseprint("The metadata file is: " + metadata)

verboseprint("Executing the solution " + algorithm)

#df_rec = SOLUTION FUNCTION
#Computing recommendation file
    
df_rec = solutions[algorithm](df_train, df_test, file_metadata = metadata, parameters = params)
# df_test = f.get_submission_target(df_test)
# df_test = df_test.rename(columns={'impressions':'item_recommendations'})
# df_test['item_recommendations'] = df_test['item_recommendations'].apply(lambda x: " ".join(x.split('|')))
# df_rec = df_test[['user_id', 'session_id', 'timestamp','step', 'item_recommendations']]
# print(df_rec.head())
# #Computing score
# algorithm = 'order_based'
subm_csv = 'submission_' + algorithm + '.csv'
if localscore == 1:
    mrr = f.score_submissions(subm_csv, gt, f.get_reciprocal_ranks)
    print("End execution with score " + str(mrr))
    file_exists = os.path.isfile('scores.csv')
    with open('scores.csv', mode='a') as score_file:
        file_writer = csv.writer(score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not file_exists: # Write headers
            file_writer.writerow(['#Epochs', '#Components', 'Loss Function', 'Learning Rate', 'Learning schedule', 'K', 'useralpha', 'itemalpha', 'rho', 'epsilon', 'maxsampled','Metadata', 'Score', 'Weight of action'])
        file_writer.writerow([str(params.epochs), str(params.ncomponents), params.lossfunction, str(params.learningrate), params.learningschedule, str(params.mfk), str(params.useralpha), str(params.itemalpha), str(params.rho), str(params.epsilon), str(params.maxsampled), str(metadata), str(mrr), json.dumps(params.actionsweights)])
    #f.send_telegram_message("End execution with score " + str(mrr))

#print(f'Mean reciprocal rank: {mrr}')
