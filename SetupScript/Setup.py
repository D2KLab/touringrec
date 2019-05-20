import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import base_solution as bs
import matrix_factorization_metadata as mf_metadata
import matrix_factorization as mf
import sys as sys
import argparse
import csv
import os.path


parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', action="store", type=str, help="Choose the algorithm that you want to use")
parser.add_argument('--train', action="store", type=str, help="--train train.csv")
parser.add_argument('--test', action="store", type=str, help="--test test.csv")
parser.add_argument('--gt', action="store", type=str, help="--gt train.csv")
parser.add_argument('--metadata', action="store", type=str, help="Define the metadata file")
parser.add_argument('--localscore', action="store", type=int, help="0 -> Local score, 1 -> Official score")
parser.add_argument('--epochs', action="store", type=int, help="Define the number of epochs")
parser.add_argument('--ncomponents', action='store', type=int, help='MF: number of components')
parser.add_argument('--lossfunction', action='store', type=str, help='MF: define the loss function')
parser.add_argument('--mfk', action='store', type=int, help='MF: parameter K')
parser.add_argument('--actions', nargs='+')




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

# Convert actions in a correct list format
list_actions = []
for i in actions:
    list_actions.append(" ".join(i.split('_')))

# Defining all the solutions implemented
solutions = {
    "basesolution": bs.get_rec_base,
    "basesolution_nation": bs.get_rec_nation,
    "matrixfactorization_metadata": mf_metadata.get_rec_matrix,
    "matrixfactorization": mf.get_rec_matrix
}

#importing and defining Datasets

#Train
print("Reading train set " + train)
df_train = pd.read_csv(train)

#Test
print("Reading test set " + test)
df_test = pd.read_csv(test)

print("Groud truth is: " + gt)

print("The metadata file is: " + metadata)

print("Executing the solution " + algorithm)

#df_rec = SOLUTION FUNCTION
#Computing recommendation file

df_rec = solutions[algorithm](df_train, df_test, file_metadata = metadata, epochs = epochs, ncomponents = ncomponents, lossfunction = lossfunction, mfk = mfk, actions = list_actions)
#Computing score
subm_csv = 'submission_' + algorithm + '.csv'
if localscore == 1:
    mrr = f.score_submissions(subm_csv, gt, f.get_reciprocal_ranks)
    print("End execution with score " + str(mrr))
    file_exists = os.path.isfile('scores.csv')
    with open('scores.csv', mode='a') as score_file:
        file_writer = csv.writer(score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not file_exists: # Write headers
            file_writer.writerow(['#Epochs', '#Components', 'Loss Function', 'K', 'List of action', 'Score'])
        file_writer.writerow([str(epochs), str(ncomponents), lossfunction, str(mfk), str(actions), str(mrr)])
    f.send_telegram_message("End execution with score " + str(mrr))


#print(f'Mean reciprocal rank: {mrr}')
