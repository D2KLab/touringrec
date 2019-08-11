# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import csv
import sys as sys
from numpy import array
from numpy import argmax
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from collections import defaultdict
from gensim.models import Word2Vec

import ds_manipulation as dsm
import test_f as tst
import LSTM as lstm
import LSTMParameters as LSTMParam

import argparse

torch.manual_seed(1)

dir = './'

parser = argparse.ArgumentParser()
parser.add_argument('--encode', action="store", type=str, help="--train encode.csv")
parser.add_argument('--meta', action="store", type=str, help="--train metadata.csv")
parser.add_argument('--traininner', action="store", type=str, help="--train train.csv")
parser.add_argument('--testinner', action="store", type=str, help="--test test.csv")
parser.add_argument('--gtinner', action="store", type=str, help="--gt train.csv", default = '')
parser.add_argument('--testdev', action="store", type=str, help="--test test.csv")
parser.add_argument('--ismeta', action='store_true', help='Use metadata')
parser.add_argument('--isimpression', action='store_true', help='Use impression list')
parser.add_argument('--isdrop', action='store_true', help='Use dropout layer')
parser.add_argument('--hiddendim', action='store', type=int, help='Set hidden dimension', default = 100)
parser.add_argument('--epochs', action="store", type=int, help="Define the number of epochs", default = 100)
parser.add_argument('--ncomponents', action='store', type=int, help='item2vec: number of components',  default = 60)
parser.add_argument('--window', action='store', type=int, help='item2vec: window length',  default = 5)
parser.add_argument('--learnrate', action='store', type=float, help='learning rate for the model', default = 0.001)
parser.add_argument('--iscuda', action='store_true', help='1 -> Use GPU, 0 -> use CPU')
parser.add_argument('--subname', action='store', type=str, help='sub file name', default='submission')
parser.add_argument('--numthread', action='store', type=int, help='sub file name', default = 1)
parser.add_argument('--batchsize', action='store', type=int, help='batch size', default = 256)
parser.add_argument('--actions', nargs='+')


# Get all the parameters
args = parser.parse_args()

param = LSTMParam.LSTMParameters(   args.encode,
                                    args.meta,
                                    args.traininner, 
                                    args.testinner,
                                    args.gtinner,
                                    args.testdev,
                                    args.ismeta,
                                    args.isimpression,
                                    args.isdrop,
                                    args.hiddendim,
                                    args.epochs,
                                    args.ncomponents,
                                    args.window,
                                    args.learnrate,
                                    args.iscuda,
                                    args.subname,
                                    args.numthread,
                                    args.batchsize)


'''
STEP 0: CONFIGURATIONS
'''

if args.iscuda:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    print('Using GPU: ' + str(device))

torch.set_num_threads(args.numthread)
number_of_threads = torch.get_num_threads()
print('Using num thread = ' + str(number_of_threads))

logfile = open(dir + param.subname + '_log' + '.txt', 'w')
logfile.write('Started ' + param.subname + ' execution\n')

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start_program_time = time.time()

'''
STEP 1: IMPORTING and MANIPULATING DATASET
'''

df_train_inner = pd.read_csv(param.traininner)
df_train_inner = dsm.remove_single_clickout_actions(df_train_inner)
df_train_inner =  dsm.remove_nonitem_actions(df_train_inner)
df_train_inner = dsm.reference_to_str(df_train_inner)

df_test_inner = pd.read_csv(param.testinner)
df_test_inner = dsm.remove_single_clickout_actions(df_test_inner)
df_test_inner = dsm.remove_nonitem_actions(df_test_inner)
df_test_for_prepare = dsm.reference_to_str(df_test_inner.copy())

if param.gtinner != '':
    df_gt_inner = pd.read_csv(param.gtinner)
else:
    df_gt_inner = ''

'''' ONLY IF WE WANT 2 TEST'''
#df_test_dev = pd.read_csv(param.testdev)
#df_test_dev = dsm.remove_single_clickout_actions(df_test_dev)
#df_test_dev = dsm.remove_nonitem_actions(df_test_dev)
#df_test_dev_for_prepare = dsm.reference_to_str(df_test_dev.copy())

#df_gt_dev = pd.read_csv('./gt_10.csv')


'''
STEP 2: PREPARE NET INPUT
'''

session_dict = {}
category_dict = {}
impression_dict = {}
session_dict, category_dict, impression_dict, train_corpus = dsm.get_training_input(df_train_inner)
print('num of sessions is ' + str(len(session_dict)) )
print('session_dict len is ' + str(len(session_dict)))
print('category_dict len is ' + str(len(category_dict)))
print('impression_dict len is ' + str(len(impression_dict)))

logfile.write('Imported and collected training set - Time: ' + str(timeSince(start_program_time)) + '\n')
print('Imported and collected training set - Time: ' + str(timeSince(start_program_time)) + '\n')
del df_train_inner

test_session_dict = {}
test_category_dict = {}
test_impression_dict = {}
test_session_dict, test_category_dict, test_impression_dict, test_corpus = dsm.get_test_input(df_test_for_prepare)
print('test_session_dict len is ' + str(len(test_session_dict)))
print('test_category_dict len is ' + str(len(test_category_dict)))
print('test_impression_dict len is ' + str(len(test_impression_dict)))

logfile.write('Imported and collected test set - Time: ' + str(timeSince(start_program_time)) + '\n')
print('Imported and collected test set - Time: ' + str(timeSince(start_program_time)) + '\n')
del df_test_for_prepare

'''' ONLY IF WE WANT 2 TEST'''
#test_dev_session_dict = {}
#test_dev_category_dict = {}
#test_dev_impression_dict = {}
#test_dev_session_dict, test_dev_category_dict, test_dev_impression_dict, test_dev_corpus = dsm.get_test_input(df_test_dev_for_prepare)
#print('test_dev_session_dict len is ' + str(len(test_dev_session_dict)))
#print('test_dev_category_dict len is ' + str(len(test_dev_category_dict)))
#print('test_dev_impression_dict len is ' + str(len(test_dev_impression_dict)))

#logfile.write('Imported and collected test dev set - Time: ' + str(timeSince(start_program_time)) + '\n')
#print('Imported and collected test dev set - Time: ' + str(timeSince(start_program_time)) + '\n')
#del test_dev_session_dict
#del test_dev_category_dict
#del test_dev_impression_dict
#del df_test_dev_for_prepare


# Batching sessions for RNN input
batched_sessions = dsm.get_batched_sessions(session_dict, category_dict, param.batchsize)
print('batched_sessions len is ' + str(len(batched_sessions)))

logfile.write('Batched trainig set - Time: ' + str(timeSince(start_program_time)) + '\n')
print(('Batched trainig set - Time: ' + str(timeSince(start_program_time)) + '\n'))

# Getting corpus for encode phase
corpus = train_corpus + test_corpus #+ test_dev_corpus

'''
STEP 3: ENCODING TO CREATE DICTIONARY
'''

#w2vec item encoding

word2vec = Word2Vec(corpus, size = param.ncomponents, min_count=1, window=param.window, sg=1)
del train_corpus
del test_corpus
#del test_dev_corpus

hotel_dict = {}
hotel_to_index_dict = {}
hotel_to_category_dict = {}

for k in word2vec.wv.index2word:
    hotel_dict[k] = torch.from_numpy(word2vec.wv[k])
    hotel_to_index_dict[k] = word2vec.wv.index2word.index(k)
    hotel_to_category_dict[k] = torch.tensor([word2vec.wv.index2word.index(k)])


del word2vec
del corpus

logfile.write('W2vec completed - Time: ' + str(timeSince(start_program_time)) + '\n')
print('W2vec completed - Time: ' + str(timeSince(start_program_time)) + '\n')

#Setting up feature numbers
n_hotels = len(hotel_dict)
n_features_w2vec = len(hotel_dict['666856'])
n_features = n_features_w2vec

print('n_hotels is ' + str(n_hotels))
print('n_features_w2vec is ' + str(n_features_w2vec))
print('n_features is ' + str(n_features))


'''
STEP 4: CREATE NETWORK
'''

#DEFINE PARAMETERS
input_dim = n_features
output_dim = n_hotels
hidden_dim = param.hiddendim
print('The model is:')
print('input_dim is:' + str(input_dim))
print('hidden_dim is: ' + str(hidden_dim))
print('output_dim is:' + str(output_dim))
layer_dim = 1

#NET CREATION
model = lstm.LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, param.iscuda)

if param.iscuda:
    model = model.cuda()


'''
STEP 5: LEARNING PHASE
'''

#LOSS FUNCTION
loss_fn = torch.nn.NLLLoss()

if param.iscuda:
    loss_fn = loss_fn.cuda()

#OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=param.learnrate)
model.optimizer = optimizer


num_epochs = param.epochs
plot_every = 1

n_iters = len(session_dict) * num_epochs
print_every = 1000

# Keep track of losses and acc for plotting
current_loss = 0
all_losses = []
all_acc = []

start = time.time()

# Training results for xgboost
training_results_hotels = {}
training_results_scores = {}

batch_hotel_window_set = []
batch_category_tensor_set = []
batch_session_tensor_set = []

timeforprep = time.time()

for batch in batched_sessions:
    max_session_len = 0
    batch_category = []
    batch_hotel_window = []
    batch_category_tensor = []

    for single_session in batch:
        if len(session_dict[single_session]) > max_session_len:
            max_session_len = len(session_dict[single_session])
        batch_category.append(category_dict[single_session])
    
    batch_category_tensor = lstm.hotels_to_category_batch(batch_category, hotel_to_category_dict, n_hotels)
    batch_session_tensor = lstm.sessions_to_batch_tensor(batch, session_dict, hotel_dict, max_session_len, n_features)

    batch_category_tensor_set.append(batch_category_tensor)
    batch_session_tensor_set.append(batch_session_tensor)

print('Got batch infos:  ' + str(timeSince(start)))

with open(dir + param.subname + 'rnn_train_inner_sub' + '.csv', mode='w') as rnn_train_sub_xgb:
    file_writer = csv.writer(rnn_train_sub_xgb)
    file_writer.writerow(['session_id', 'hotel_id', 'score'])

    df_train_inner_sub_list = []

    for epoch in range(1, num_epochs + 1):

        logfile.write('Epoch ' + str(epoch) + ' start - Time: ' + str(timeSince(start)) + '\n')

        iter = 0
        
        count_correct = 0
        count_correct_windowed = 0
        
        for batch_i, batch in enumerate(batched_sessions):
            iter = iter + 1

            batch_category_tensor = batch_category_tensor_set[batch_i]
            batch_session_tensor = batch_session_tensor_set[batch_i]

            output, loss = lstm.train(model, loss_fn, optimizer, batch_category_tensor, batch_session_tensor, param.iscuda)

            current_loss += loss
            
            '''
            if epoch == num_epochs:
                guess_windowed_list, guess_windowed_scores_list = lstm.categories_from_output_windowed_opt(output, batch, impression_dict, hotel_dict, hotel_to_index_dict, df_train_inner_sub_list, pickfirst = False)
        
                for batch_i, single_session in enumerate(batch):
                    for hotel_i, hotel in enumerate(guess_windowed_list[batch_i]):
                        # Write single hotel score
                        file_writer.writerow([str(single_session), str(hotel), str(guess_windowed_scores_list[batch_i][hotel_i])])
            '''
                
        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / (plot_every * len(batched_sessions)))
            print('Epoch: ' + str(epoch) + ' Loss: ' + str(current_loss / (plot_every * len(batched_sessions))))
            print('%d %d%% (%s)' % (epoch, epoch / num_epochs * 100, timeSince(start)))
            current_loss = 0

        if epoch == num_epochs + 1:
            torch.save(model.state_dict(), dir + param.subname + '_model_epoch_' + str(epoch))

        logfile.write('Epoch ' + str(epoch) + ' end - Time: ' + str(timeSince(start)) + '\n')
        logfile.write('Epoch ' + str(epoch) + ' - Loss: ' + str(all_losses[-1]) + '\n')
        

del batch_session_tensor_set
del batch_category_tensor_set


'''
STEP 6: PLOTTING RESULTS
'''

#plt.figure()
#plt.plot(all_losses)

#plt.figure()
#plt.plot(all_acc)


'''
STEP 7: Save Test Results
'''

start_test_time = time.time()

logfile.write('Start submission - Time: ' + str(timeSince(start_test_time)) + '\n')

mrr = tst.test_accuracy_optimized_classification(model, df_test_inner, df_gt_inner, test_session_dict, test_category_dict, test_impression_dict, hotel_dict, hotel_to_index_dict, n_features, dir, param.subname, isprint=True, dev = False)
print("Final score: " + str(mrr))
print(timeSince(start_test_time))

logfile.write('Finish submission - Time: ' + str(timeSince(start_test_time)) + '\n')

''' Only if we want 2 tests'''
#logfile.write('Start dev submission - Time: ' + str(timeSince(start_test_time)) + '\n')

#mrr = tst.test_accuracy_optimized_classification(model, df_test_dev, df_gt_inner, test_session_dict, test_category_dict, test_impression_dict, hotel_dict, hotel_to_index_dict, n_features, param.subname, isprint=True, dev = True)
#print("Final score for dev: " + str(mrr))
#print(timeSince(start_test_time))

#logfile.write('Finish dev submission - Time: ' + str(timeSince(start_test_time)) + '\n')

'''
STEP 8: SAVING SUBMISSION
'''

#Saving loss
with open(dir + param.subname + '_loss.csv', mode='w') as loss_file:
    file_writer = csv.writer(loss_file)
    file_writer.writerow(['#Epochs'])
    for loss in all_losses:
        file_writer.writerow([loss])
'''
#Saving acc
with open(param.subname + '_acc.csv', mode='w') as acc_file:
    file_writer = csv.writer(acc_file)
    file_writer.writerow(['#Epochs'])
    for acc in all_acc:
        file_writer.writerow([acc])
'''