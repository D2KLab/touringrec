import pandas as pd
import numpy as np
import functions as f
import mf_xgboost as mf_xg
import sys as sys
import argparse
import csv
import os.path
import json
from MFParameters import MF_Parameters

parser = argparse.ArgumentParser()
parser.add_argument('--outputfile', action="store", type=str, help="Choose the filename of the output file", default='submission_mfxgboost.csv')
parser.add_argument('--train', action="store", type=str, help="--train train.csv", default='train.csv')
parser.add_argument('--test', action="store", type=str, help="--test test.csv", default='test.csv')
parser.add_argument('--traininner', action="store", type=str, help="--traininner train_inner.csv", default='train_inner.csv"')
parser.add_argument('--gtinner', action="store", type=str, help="--testinner gt_inner.csv", default='gt_inner.csv')
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
subm_csv = args.outputfile
train = args.train
test = args.test
gt = args.gt
traininner = args.traininner
gtinner = args.gtinner
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

params = MF_Parameters(epochs, ncomponents, lossfunction, mfk, learningrate, learningschedule, useralpha, itemalpha, rho, epsilon, maxsampled, actions_weights, list_actions)
df_train = pd.read_csv(train)
df_test = pd.read_csv(test)

df_rec = mf_xg.get_rec_matrix(df_train, df_test, traininner, gtinner, subm_csv, file_metadata = metadata, parameters = params)
df_test = f.get_submission_target(df_test)

if localscore == 1:
    mrr = f.score_submissions(subm_csv, gt, f.get_reciprocal_ranks)
    print("End execution with score " + str(mrr))