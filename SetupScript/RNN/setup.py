# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import glob
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

import argparse

torch.manual_seed(1)

parser = argparse.ArgumentParser()
#parser.add_argument('--algorithm', action="store", type=str, help="Choose the algorithm that you want to use")
parser.add_argument('--train', action="store", type=str, help="--train train.csv")
parser.add_argument('--test', action="store", type=str, help="--test test.csv")
parser.add_argument('--gt', action="store", type=str, help="--gt train.csv")
#parser.add_argument('--metadata', action="store", type=str, help="Define the metadata file")
#parser.add_argument('--localscore', action="store", type=int, help="0 -> Local score, 1 -> Official score")
parser.add_argument('--epochs', action="store", type=int, help="Define the number of epochs")
parser.add_argument('--ncomponents', action='store', type=int, help='item2vec: number of components')
#parser.add_argument('--lossfunction', action='store', type=str, help='MF: define the loss function')
parser.add_argument('--window', action='store', type=int, help='item2vec: window length')
parser.add_argument('--learnrate', action='store', type=int, help='learning rate for the model')
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
window = args.window
learning_rate = args.learnrate
actions = args.actions


print("Reading train set " + train)
print("Reading test set " + test)
print("Groud truth is: " + gt)
#print("The metadata file is: " + metadata)
#print("Executing the solution " + algorithm)


'''
STEP 1: IMPORTING and MANIPULATING DATASET
'''

#importing encode set
df_encode = pd.read_csv(train)
df_encode = dsm.remove_single_actions(df_encode)
df_encode = dsm.remove_nonitem_actions(df_encode)
#df_encode = dsm.reduce_df(df_encode, 80000)

#importing training set
df_train = pd.read_csv(train)
df_train = dsm.remove_single_actions(df_train)
df_train =  dsm.remove_nonitem_actions(df_train)
#df_train = dsm.reduce_df(df_train, 10000)

#importing test set
df_test = pd.read_csv(test)
df_test = dsm.remove_single_actions(df_test)
df_test = dsm.remove_nonitem_actions(df_test)
#df_test = dsm.reduce_df(df_test, 1000)

#importing ground truth
df_gt = pd.read_csv(gt)
#df_gt = dsm.reduce_df(df_gt, 1000)

df_test, df_gt = dsm.remove_test_single_actions(df_test, df_gt)

corpus = dsm.get_corpus(df_encode)


'''
STEP 2: ENCODING TO CREATE DICTIONARY
'''

#w2vec item encoding
from gensim.models import Word2Vec

word2vec = Word2Vec(corpus, min_count=1, window=window, sg=1)

n_features = len(word2vec.wv['666856'])

#hotel_dict = w2v.normalize_word2vec(word2vec.wv)

hotel_dict = word2vec.wv

n_hotels = len(hotel_dict.index2word)
n_features = len(word2vec.wv['666856'])

print('n_hotels is ' + str(n_hotels))
print('n_features is ' + str(n_features))


'''
STEP 3: PREPARE NET INPUT
'''

#this splits the training set sessions into multiple mini-sessions
sessions, categories, hotels_window = dsm.prepare_input(df_train)


'''
STEP 4: CREATE NETWORK
'''

input_dim = n_features
output_dim = n_hotels
hidden_dim = int(1/3 * (input_dim + output_dim))
print('hidden_dim is ' + str(hidden_dim))
layer_dim = 1 #try more hidden layers

#NET CREATION
model = lstm.LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).cuda()


'''
STEP 5: LEARNING PHASE
'''

loss_fn = torch.nn.CrossEntropyLoss().cuda()

learning_rate = learning_rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

import time
import math


num_epochs = epochs
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
  model.train()
  iter = 0
  
  print(str(len(sessions)) + ' sessions to be computed')
  
  for index, session in enumerate(sessions):
    iter = iter + 1

    session_tensor = lstm.session_to_tensor(session)
    category = categories[index]
    category_tensor = lstm.hotel_to_category(category, hotel_dict, n_hotels)

    
    output, loss = lstm.train(category_tensor, session_tensor)

    current_loss += loss
      
    if iter % print_every == 0:
        guess, guess_i = lstm.category_from_output(output)

        correct = '✓' if guess == category else '✗ (%s)' % category
        print('(%s) %.4f %s / %s %s' % (timeSince(start), loss, session[0]['session_id'], guess, correct))

        
  # Add current loss avg to list of losses
  if epoch % plot_every == 0:
      all_losses.append(current_loss / (plot_every * len(sessions)))
      print('Epoch: ' + str(epoch) + ' Loss: ' + str(current_loss / (plot_every * len(sessions))))
      print('%d %d%% (%s)' % (epoch, epoch / num_epochs * 100, timeSince(start)))
      acc = tst.test_accuracy(model, df_test, df_gt)
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

mrr = tst.test_accuracy(model, df_test, df_gt)
print("Final score: " + str(mrr))


'''
STEP 8: SAVING SUBMISSION
'''

#Computing score
subm_csv = 'submission.csv'

print("End execution with score " + str(mrr))
file_exists = os.path.isfile('scores.csv')
with open('scores.csv', mode='a') as score_file:
    file_writer = csv.writer(score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    if not file_exists: # Write headers
        file_writer.writerow(['#Epochs', '#Components', 'W2Vec window', 'Learn Rate', 'Score'])
    file_writer.writerow([str(epochs), str(ncomponents), str(window), str(learning_rate), str(mrr)])
f.send_telegram_message("End execution with score " + str(mrr))