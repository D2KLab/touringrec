# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
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

import ds_manipulation as dsm
import w2vec as w2v
import test_f as tst
import LSTM as lstm
import LSTMParameters as LSTMParam

import argparse

#python3 python3 setup.py --train ./train_1.csv --test ./test_1.csv --gt ./gt_1.csv --epochs 10 --ncomponents 100 --window 3 --learnrate 0.001
torch.manual_seed(1)

parser = argparse.ArgumentParser()
#parser.add_argument('--algorithm', action="store", type=str, help="Choose the algorithm that you want to use")
parser.add_argument('--encode', action="store", type=str, help="--train encode.csv")
parser.add_argument('--meta', action="store", type=str, help="--train metadata.csv")
parser.add_argument('--train', action="store", type=str, help="--train train.csv")
parser.add_argument('--test', action="store", type=str, help="--test test.csv")
parser.add_argument('--gt', action="store", type=str, help="--gt train.csv")
#parser.add_argument('--metadata', action="store", type=str, help="Define the metadata file")
#parser.add_argument('--localscore', action="store", type=int, help="0 -> Local score, 1 -> Official score")
parser.add_argument('--ismeta', action='store_true', help='Use metadata')
parser.add_argument('--isimpression', action='store_true', help='Use impression list')
parser.add_argument('--isdrop', action='store_true', help='Use dropout layer')
parser.add_argument('--hiddendim', action='store', type=int, help='Set hidden dimension')
parser.add_argument('--epochs', action="store", type=int, help="Define the number of epochs")
parser.add_argument('--ncomponents', action='store', type=int, help='item2vec: number of components')
#parser.add_argument('--lossfunction', action='store', type=str, help='MF: define the loss function')
parser.add_argument('--window', action='store', type=int, help='item2vec: window length')
parser.add_argument('--learnrate', action='store', type=float, help='learning rate for the model')
parser.add_argument('--iscuda', action='store_true', help='1 -> Use GPU, 0 -> use CPU')
parser.add_argument('--subname', action='store', type=str, help='sub file name', default='submission')
parser.add_argument('--numthread', action='store', type=int, help='sub file name', default=1)
parser.add_argument('--batchsize', action='store', type=int, help='batch size', default=0)
parser.add_argument('--actions', nargs='+')


# Get all the parameters
args = parser.parse_args()

param = LSTMParam.LSTMParameters(   args.encode,
                                    args.meta,
                                    args.train, 
                                    args.test,
                                    args.gt,
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


#print("Reading train set " + param.train)
#print("Reading test set " + param.test)
#print("Groud truth is: " + param.gt)
#print("The metadata file is: " + metadata)
#print("Executing the solution " + algorithm)


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


'''
STEP 1: IMPORTING and MANIPULATING DATASET
'''

#importing encode set
df_encode = pd.read_csv(param.encode)
df_encode = dsm.remove_single_actions(df_encode)
df_encode = dsm.remove_nonitem_actions(df_encode)
#df_encode = dsm.reduce_df(df_encode, 10000)

#importing metadata set
df_meta = []
if param.ismeta:
    df_meta = pd.read_csv(param.meta)

#importing training set
df_train = pd.read_csv(param.train)
df_train = dsm.remove_single_actions(df_train)
df_train =  dsm.remove_nonitem_actions(df_train)
#df_train = dsm.reduce_df(df_train, 100)

#importing test set
df_test = pd.read_csv(param.test)
df_test = dsm.remove_single_actions(df_test)
df_test = dsm.remove_nonitem_actions(df_test)
#df_test = dsm.reduce_df(df_test, 100)

#importing ground truth
df_gt = pd.read_csv(param.gt)
#df_gt = dsm.reduce_df(df_gt, 100)

df_test, df_gt = dsm.remove_test_single_actions(df_test, df_gt)

corpus = dsm.get_corpus(df_encode)


'''
STEP 2: ENCODING TO CREATE DICTIONARY
'''

#w2vec item encoding
from gensim.models import Word2Vec

word2vec = Word2Vec(corpus, min_count=1, window=param.window, sg=1)

n_features = len(word2vec.wv['666856'])

#hotel_dict = w2v.normalize_word2vec(word2vec.wv)

hotel_dict = word2vec.wv

#extracting metadata features
meta_list = []
meta_dict = []
if param.ismeta:
    meta_list = dsm.extract_unique_meta(df_meta)
    meta_dict = dsm.get_meta_dict(df_meta, hotel_dict.index2word, meta_list)


'''
STEP 3: PREPARE NET INPUT
'''

#this splits the training set sessions into multiple mini-sessions
if param.batchsize == 0:
    sessions, categories, hotels_window = dsm.prepare_input(df_train)
else:
    sessions, categories, hotels_window = dsm.prepare_input_batched(df_train, param.batchsize)

test_sessions, test_hotels_window, test_clickout_index = tst.prepare_test(df_test, df_gt)

#getting maximum window size
max_window = 0
if param.isimpression:
    for window in hotels_window:
        if len(window) > max_window:
            max_window = len(window)
    if param.train == './train_1.csv':
        max_window = 25

#Setting up feature numbers
n_hotels = len(hotel_dict.index2word)
n_features_w2vec = len(word2vec.wv['666856'])
n_features_meta = len(meta_list)
n_features_impression = max_window
n_features = n_features_w2vec + n_features_meta + n_features_impression

print('n_hotels is ' + str(n_hotels))
print('n_features_w2vec is ' + str(n_features_w2vec))
print('n_features_meta is ' + str(n_features_meta))
print('n_features_impression is ' + str(n_features_impression))
print('n_features is ' + str(n_features))


'''
STEP 4: CREATE NETWORK
'''

#DEFINE PARAMETERS
input_dim = n_features
output_dim = n_hotels
#hidden_dim = int(1/100 * (input_dim + output_dim))
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

#WEIGHT INIT
model.lstm.weight_hh_l0.data.fill_(0)
x = 1
nn.init.uniform_(model.fc.weight, -x, x)
nn.init.uniform_(model.fc.bias, -x, x)

'''
STEP 5: LEARNING PHASE
'''

#LOSS FUNCTION
loss_fn = torch.nn.CrossEntropyLoss()

if param.iscuda:
    loss_fn = loss_fn.cuda()

#OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=param.learnrate)
model.optimizer = optimizer

import time
import math


num_epochs = param.epochs
plot_every = 1

n_iters = len(sessions) * num_epochs
print_every = 100

# Keep track of losses and acc for plotting
current_loss = 0
all_losses = []
all_acc = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, num_epochs + 1):
  #model.train()
  iter = 0
  
  print(str(len(sessions)) + ' sessions to be computed')
  
  for index, session in enumerate(sessions):
    iter = iter + 1

    if param.batchsize == 0:
        session_tensor = lstm.session_to_tensor(session, hotel_dict, n_features, hotels_window, max_window, meta_dict, meta_list)
        category = categories[index]
        category_tensor = lstm.hotel_to_category(category, hotel_dict, n_hotels)
    else:
      max_session_len = 0
      for si, single_session in enumerate(session):
        if len(single_session) > max_session_len:
          max_session_len = len(single_session)
          
      session_tensor = lstm.sessions_to_batch(session, hotel_dict, max_session_len, n_features, hotels_window, max_window, meta_dict, meta_list)
      category = categories[index]
      category_tensor = lstm.hotels_to_category_batch(category, hotel_dict, n_hotels)

    
    output, loss = lstm.train(model, loss_fn, optimizer, category_tensor, session_tensor, param.iscuda)

    current_loss += loss
      
    if iter % print_every == 0:
        guess, guess_i = lstm.category_from_output(output, hotel_dict)

        correct = '✓' if guess == category else '✗ (%s)' % category
        #print('(%s) %.4f %s / %s %s' % (timeSince(start), loss, session[0]['session_id'], guess, correct))

        
  # Add current loss avg to list of losses
  if epoch % plot_every == 0:
      all_losses.append(current_loss / (plot_every * len(sessions)))
      print('Epoch: ' + str(epoch) + ' Loss: ' + str(current_loss / (plot_every * len(sessions))))
      print('%d %d%% (%s)' % (epoch, epoch / num_epochs * 100, timeSince(start)))
      #acc = tst.test_accuracy(model, df_test, df_gt, hotel_dict, n_features, max_window, meta_dict, meta_list)
      acc = tst.test_accuracy_optimized(model, df_test, df_gt, test_sessions, test_hotels_window, test_clickout_index, hotel_dict, n_features, max_window, meta_dict, meta_list)
      print("Score: " + str(acc))
      all_acc.append(acc)
      current_loss = 0


'''
STEP 6: PLOTTING RESULTS
'''

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

plt.figure()
plt.plot(all_acc)


'''
STEP 7: PREPARE TEST SET
'''

#mrr = tst.test_accuracy(model, df_test, df_gt, hotel_dict, n_features, max_window, meta_dict, meta_list, param.subname, isprint=True)
mrr = tst.test_accuracy_optimized(model, df_test, df_gt, test_sessions, test_hotels_window, test_clickout_index, hotel_dict, n_features, max_window, meta_dict, meta_list, param.subname, isprint=True)
print("Final score: " + str(mrr))


'''
STEP 8: SAVING SUBMISSION
'''

#Computing score
#print("End execution with score " + str(mrr))
file_exists = os.path.isfile('scores.csv')
with open('scores.csv', mode='a') as score_file:
    file_writer = csv.writer(score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    if not file_exists: # Write headers
        file_writer.writerow(['Train set', 'Using impressions', 'Using meta', 'Hidden dimension', 'Dropout layer', '#Epochs', '#Components', 'W2Vec window', 'Learn Rate', 'batchsize' 'Score'])
    file_writer.writerow([str(param.train), str(param.isimpression), str(param.ismeta), str(param.hiddendim), str(param.isdrop), str(param.epochs), str(param.ncomponents), str(param.window), str(param.learnrate), str(param.batchsize), str(mrr)])
#f.send_telegram_message("End execution with score " + str(mrr))

#Saving loss
with open(param.subname + '_loss.csv', mode='w') as loss_file:
    file_writer = csv.writer(loss_file)
    file_writer.writerow(['#Epochs'])
    for loss in all_losses:
        file_writer.writerow([loss])

#Saving acc
with open(param.subname + '_acc.csv', mode='a') as acc_file:
    file_writer = csv.writer(acc_file)
    file_writer.writerow(['#Epochs'])
    for acc in all_acc:
        file_writer.writerow([acc])