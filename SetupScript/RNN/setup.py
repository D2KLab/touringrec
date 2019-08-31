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

import ds_manipulation as dsm
import w2vec as w2v
import test_f as tst
import LSTM as lstm
import LSTMParameters as LSTMParam

import argparse

#python3 python3 setup.py --train ./train_1.csv --test ./test_1.csv --gt ./gt_1.csv --epochs 10 --ncomponents 100 --window 3 --learnrate 0.001
#python3 setup.py --encode ./encode_1.csv --meta ./item_metadata.csv --train ./train_1.csv --test ./test_1.csv --gt ./gt_1.csv --hiddendim 100 --epochs 20 --ncomponents 100 --window 5 --learnrate 0.001 --iscuda --subname rnn_1%_sub --numthread 2 --batchsize 16
torch.manual_seed(1)

dir = './'

parser = argparse.ArgumentParser()
#parser.add_argument('--algorithm', action="store", type=str, help="Choose the algorithm that you want to use")
parser.add_argument('--encode', action="store", type=str, help="--train encode.csv")
parser.add_argument('--meta', action="store", type=str, help="--train metadata.csv")
parser.add_argument('--traininner', action="store", type=str, help="--train train.csv")
parser.add_argument('--testinner', action="store", type=str, help="--test test.csv")
parser.add_argument('--gtinner', action="store", type=str, help="--gt train.csv")
parser.add_argument('--testdev', action="store", type=str, help="--test test.csv")
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

#importing encode set
#df_encode = pd.read_csv(param.encode)
#df_encode = dsm.remove_single_actions(df_encode)
#df_encode = dsm.remove_nonitem_actions(df_encode)
#df_encode = dsm.reduce_df(df_encode, 10000)

#importing metadata set
df_meta = []
if param.ismeta:
    df_meta = pd.read_csv(param.meta)

#importing training set
#df_train = pd.read_csv(param.train)
#df_train = dsm.remove_single_actions(df_train)
#df_train =  dsm.remove_nonitem_actions(df_train)
#df_train = dsm.reduce_df(df_train, 100)

#importing test set
#df_test = pd.read_csv(param.test)
#df_test = dsm.remove_single_actions(df_test)
#df_test = dsm.remove_nonitem_actions(df_test)
#df_test = dsm.reduce_df(df_test, 100)

#importing ground truth
#df_gt = pd.read_csv(param.gt)
#df_gt = dsm.reduce_df(df_gt, 100)

#df_test, df_gt = dsm.remove_test_single_actions(df_test, df_gt)

#corpus = dsm.get_corpus(df_encode)

df_train_inner = pd.read_csv(param.traininner)
df_train_inner = dsm.remove_single_clickout_actions(df_train_inner)
#df_train_inner = dsm.remove_single_actions_opt(df_train_inner)
df_train_inner =  dsm.remove_nonitem_actions(df_train_inner)
df_train_inner = dsm.reference_to_str(df_train_inner)

df_test_inner = pd.read_csv(param.testinner)
df_test_inner = dsm.remove_single_clickout_actions(df_test_inner)
#df_test_inner = dsm.remove_single_actions_opt(df_test_inner)
df_test_inner = dsm.remove_nonitem_actions(df_test_inner)
df_test_for_prepare = dsm.reference_to_str(df_test_inner.copy())

# No need for gt
#df_gt_inner = []
df_gt_inner = pd.read_csv(param.gtinner)
#df_gt_inner = dsm.remove_single_clickout_actions(df_gt_inner)

#df_test_inner, df_gt_inner = dsm.remove_test_single_actions(df_test_inner, df_gt_inner)

'''
df_test_dev = pd.read_csv(param.testdev)
df_test_dev = dsm.remove_single_clickout_actions(df_test_dev)
#df_test_dev = dsm.remove_single_actions_opt(df_test_dev)
df_test_dev = dsm.remove_nonitem_actions(df_test_dev)
df_test_dev_for_prepare = dsm.reference_to_str(df_test_dev.copy())
'''

#df_gt_dev = pd.read_csv('./gt_10.csv')

#df_test_inner, df_gt_inner = dsm.remove_test_single_actions(df_test_dev, df_gt_dev)


'''
STEP 2: PREPARE NET INPUT
'''

#this splits the training set sessions into multiple mini-sessions
'''
if param.batchsize == 0:
    sessions, categories, hotels_window = dsm.prepare_input(df_train_inner)
else:
    sessions, categories, hotels_window = dsm.prepare_input_batched(df_train_inner, param.batchsize)
'''

#test_sessions, test_hotels_window, test_clickout_index = tst.prepare_test(df_test_inner, df_gt_inner)

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
del df_test_for_prepare

'''
df_test_inner = pd.read_csv(param.testinner)
df_test_inner = dsm.remove_single_clickout_actions(df_test_inner)
df_test_inner = dsm.remove_nonitem_actions(df_test_inner)
'''

logfile.write('Imported and collected test set - Time: ' + str(timeSince(start_program_time)) + '\n')
print('Imported and collected test set - Time: ' + str(timeSince(start_program_time)) + '\n')

'''
test_dev_session_dict = {}
test_dev_category_dict = {}
test_dev_impression_dict = {}
test_dev_session_dict, test_dev_category_dict, test_dev_impression_dict, test_dev_corpus = dsm.get_test_input(df_test_dev_for_prepare)
print('test_dev_session_dict len is ' + str(len(test_dev_session_dict)))
print('test_dev_category_dict len is ' + str(len(test_dev_category_dict)))
print('test_dev_impression_dict len is ' + str(len(test_dev_impression_dict)))
del test_dev_session_dict
del test_dev_category_dict
del test_dev_impression_dict
del df_test_dev_for_prepare
'''

logfile.write('Imported and collected test dev set - Time: ' + str(timeSince(start_program_time)) + '\n')
print('Imported and collected test dev set - Time: ' + str(timeSince(start_program_time)) + '\n')

# Batching sessions for RNN input
batched_sessions = dsm.get_batched_sessions(session_dict, category_dict, param.batchsize)
print('batched_sessions len is ' + str(len(batched_sessions)))

logfile.write('Batched trainig set - Time: ' + str(timeSince(start_program_time)) + '\n')
print(('Batched trainig set - Time: ' + str(timeSince(start_program_time)) + '\n'))

#df_corpus = pd.concat([df_train_inner, df_test_inner, df_test_dev], ignore_index=True)
#df_corpus = dsm.reference_to_str(df_corpus)
corpus = train_corpus + test_corpus + test_dev_corpus

'''
STEP 3: ENCODING TO CREATE DICTIONARY
'''

#w2vec item encoding
from gensim.models import Word2Vec

word2vec = Word2Vec(corpus, size = param.ncomponents, min_count=1, window=param.window, sg=1)
del train_corpus
del test_corpus
del test_dev_corpus


n_features = len(word2vec.wv['666856'])
#print(type(word2vec.wv.vocab.items()))
#hotel_dict = w2v.normalize_word2vec(word2vec.wv)

hotel_dict = {}
hotel_to_index_dict = {}
hotel_to_category_dict = {}

'''
def populate_dict(k, word2vec, hotel_to_index_dict, hotel_to_category_dict):
    global hotel_dict
    hotel_dict[k] = torch.from_numpy(word2vec.wv[k])
    hotel_to_index_dict[k] = word2vec.wv.index2word.index(k)
    hotel_to_category_dict[k] = torch.tensor([word2vec.wv.index2word.index(k)])
    return

map(lambda k: populate_dict(k, word2vec, hotel_to_index_dict, hotel_to_category_dict), word2vec.wv.index2word)
'''

for k in word2vec.wv.index2word:
    hotel_dict[k] = torch.from_numpy(word2vec.wv[k])
    hotel_to_index_dict[k] = word2vec.wv.index2word.index(k)
    hotel_to_category_dict[k] = torch.tensor([word2vec.wv.index2word.index(k)])


#print(hotel_dict)

'''
hotel_dict = {k:torch.from_numpy(word2vec.wv[k]) for k in word2vec.wv.index2word}
hotel_to_index_dict = {k: list(hotel_dict.keys()).index(k) for k in hotel_dict}
hotel_to_category_dict = {k:torch.tensor([list(hotel_dict.keys()).index(k)]) for k in hotel_dict}
'''
del word2vec
del corpus

logfile.write('W2vec completed - Time: ' + str(timeSince(start_program_time)) + '\n')
print('W2vec completed - Time: ' + str(timeSince(start_program_time)) + '\n')
#extracting metadata features
meta_list = []
meta_dict = []
if param.ismeta:
    meta_list = dsm.extract_unique_meta(df_meta)
    meta_dict = dsm.get_meta_dict(df_meta, hotel_dict, meta_list)


#getting maximum window size
max_window = 0
'''
if param.isimpression:
    for window in hotels_window:
        if len(window) > max_window:
            max_window = len(window)
    #if param.train == './train_1.csv':
    max_window = 25
'''

#Setting up feature numbers
n_hotels = len(hotel_dict)
n_features_w2vec = len(hotel_dict['666856'])
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

# WEIGHT INIT LSTM
#model.lstm.weight_hh_l0.data.fill_(0)
#x = 1
#nn.init.uniform_(model.fc.weight, -x, x)
#nn.init.uniform_(model.fc.bias, -x, x)

# WEIGHT INIT GRU???
# TODO

'''
STEP 5: LEARNING PHASE
'''

#LOSS FUNCTION
#loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.NLLLoss()

if param.iscuda:
    loss_fn = loss_fn.cuda()

#OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=param.learnrate)
model.optimizer = optimizer

import time
import math


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

#max_session_len_set = []
#batch_category_set = []
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
        #print('start max_session_len in time ' + str(timeSince(timeforprep)))
        if len(session_dict[single_session]) > max_session_len:
            max_session_len = len(session_dict[single_session])
        #print('fninsh max_session_len in time ' + str(timeSince(timeforprep)))
        #print('start batch_category in time ' + str(timeSince(timeforprep)))
        batch_category.append(category_dict[single_session])
        #print('finish batch_category in time ' + str(timeSince(timeforprep)))
        #batch_hotel_window.append(impression_dict[single_session])
    
    #print('start batch_category_tensor in time ' + str(timeSince(timeforprep)))
    batch_category_tensor = lstm.hotels_to_category_batch(batch_category, hotel_to_category_dict, n_hotels)
    #print('finish batch_category_tensor in time ' + str(timeSince(timeforprep)))

    #print('start session tensor in time ' + str(timeSince(timeforprep)))
    batch_session_tensor = lstm.sessions_to_batch_tensor(batch, session_dict, hotel_dict, max_session_len, n_features)
    #print('finish session tensor in time ' + str(timeSince(timeforprep)))

    #max_session_len_set.append(max_session_len)
    #batch_category_set.append(batch_category)
    #batch_hotel_window_set.append(batch_hotel_window)
    batch_category_tensor_set.append(batch_category_tensor)
    batch_session_tensor_set.append(batch_session_tensor)
    #print('Finished batch prep in time ' + str(timeSince(timeforprep)))

print('Got batch infos:  ' + str(timeSince(start)))

with open(dir + param.subname + 'rnn_train_inner_sub' + '.csv', mode='w') as rnn_train_sub_xgb:
    file_writer = csv.writer(rnn_train_sub_xgb)
    file_writer.writerow(['session_id', 'hotel_id', 'score'])

    df_train_inner_sub_list = []

    for epoch in range(1, num_epochs + 1):

        logfile.write('Epoch ' + str(epoch) + ' start - Time: ' + str(timeSince(start)) + '\n')

        #model.train()
        iter = 0
        
        count_correct = 0
        count_correct_windowed = 0

        #print(str(len(session_dict)) + ' sessions to be computed')
        
        for batch_i, batch in enumerate(batched_sessions):
            iter = iter + 1

            '''
            max_session_len = 0
            batch_category = []
            batch_hotel_window = []
            for si, single_session in enumerate(batch):
                if len(session_dict[single_session]) > max_session_len:
                    max_session_len = len(session_dict[single_session])
                batch_category.append(category_dict[single_session])
                batch_hotel_window.append(impression_dict[single_session])
                batch_category_tensor = lstm.hotels_to_category_batch(batch_category, hotel_dict, n_hotels)
            

            batch_session_tensor = lstm.sessions_to_batch_tensor(batch, session_dict, hotel_dict, max_session_len, n_features)

            '''
            #max_session_len = max_session_len_set[batch_i]
            #batch_category = batch_category_set[batch_i]
            #batch_hotel_window = batch_hotel_window_set[batch_i]
            batch_category_tensor = batch_category_tensor_set[batch_i]
            
            batch_session_tensor = batch_session_tensor_set[batch_i]
            #print('Turned session to batch : ' + str(timeSince(start)))

            output, loss = lstm.train(model, loss_fn, optimizer, batch_category_tensor, batch_session_tensor, param.iscuda)

            current_loss += loss
            
            if epoch == num_epochs:
                guess_windowed_list, guess_windowed_scores_list = lstm.categories_from_output_windowed_opt(output, batch, impression_dict, hotel_dict, hotel_to_index_dict, df_train_inner_sub_list, pickfirst = False)
        
                for batch_i, single_session in enumerate(batch):
                    #if guess[batch_i] == category_v:
                    #    count_correct = count_correct + 1

                    #if iter % print_every == 0:
                        #print('Non-Windowed results:')
                        #correct = '✓' if guess[batch_i] == category_v else '✗ (%s)' % category_v
                        #print('(%s) %.4f %s / %s %s' % (timeSince(start), loss, session[batch_i][0]['session_id'], guess[batch_i], correct))

                    #if guess_windowed_list[batch_i][0] == category_v:
                    #    count_correct_windowed = count_correct_windowed + 1

                    #if iter % print_every == 0:
                        #print('Windowed results:')
                        #correct = '✓' if guess_windowed_list[batch_i][0] == category_v else '✗ (%s)' % category_v
                        #print('(%s) %.4f %s / %s %s' % (timeSince(start), loss, session[batch_i][0]['session_id'], guess_windowed_list[batch_i][0], correct))

                    for hotel_i, hotel in enumerate(guess_windowed_list[batch_i]):
                        # Write single hotel score
                        file_writer.writerow([str(single_session), str(hotel), str(guess_windowed_scores_list[batch_i][hotel_i])])
                    
                
        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / (plot_every * len(batched_sessions)))
            print('Epoch: ' + str(epoch) + ' Loss: ' + str(current_loss / (plot_every * len(batched_sessions))))
            print('%d %d%% (%s)' % (epoch, epoch / num_epochs * 100, timeSince(start)))
            #print('Found ' + str(count_correct) + ' correct clickouts among ' + str(len(sessions) * param.batchsize) + ' sessions.')
            #print('Windowed - Found ' + str(count_correct_windowed) + ' correct clickouts among ' + str(len(sessions) * param.batchsize) + ' sessions.')
            #acc = tst.test_accuracy_optimized(model, df_test_inner, df_gt_inner, test_sessions, test_hotels_window, test_clickout_index, hotel_dict, n_features, max_window, meta_dict, meta_list)
            #print("Score: " + str(acc))
            #all_acc.append(acc)
            current_loss = 0

        if epoch % 10 == 0:
            torch.save(model.state_dict(), dir + param.subname + 'model_epoch_' + str(epoch))

        logfile.write('Epoch ' + str(epoch) + ' end - Time: ' + str(timeSince(start)) + '\n')
        logfile.write('Epoch ' + str(epoch) + ' - Loss: ' + str(all_losses[-1]) + '\n')
        

del batch_session_tensor_set
del batch_category_tensor_set

# Print df_train_inner
#df_train_inner_sub = pd.DataFrame(df_train_inner_sub_list, columns = ['session_id', 'hotel_id', 'score'])
#df_train_inner_sub = df_train_inner_sub.groupby('session_id').apply(lambda x: x.sort_values(['score'], ascending = False))
#df_train_inner_sub.to_csv(dir + 'rnn_train_inner_sub' + param.subname + '.csv')


'''
STEP 6: PLOTTING RESULTS
'''

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#plt.figure()
#plt.plot(all_losses)

#plt.figure()
#plt.plot(all_acc)


'''
STEP 7: Save Test Results
'''

start_test_time = time.time()

logfile.write('Start inner submission - Time: ' + str(timeSince(start_test_time)) + '\n')

#mrr = tst.test_accuracy(model, df_test, df_gt, hotel_dict, n_features, max_window, meta_dict, meta_list, param.subname, isprint=True)
mrr = tst.test_accuracy_optimized_classification(model, df_test_inner, df_gt_inner, test_session_dict, test_category_dict, test_impression_dict, hotel_dict, hotel_to_index_dict, n_features, max_window, meta_dict, meta_list, dir, param.subname, isprint=True, dev = False)
print("Final score for inner: " + str(mrr))
print(timeSince(start_test_time))

logfile.write('Finish inner submission - Time: ' + str(timeSince(start_test_time)) + '\n')

#test_sessions, test_hotels_window, test_clickout_index = tst.prepare_test(df_test_dev, df_gt_dev)

# Done above
#test_session_dict, test_category_dict, test_impression_dict, test_corpus = dsm.get_test_input(df_test_dev_for_prepare)

'''
df_test_dev = pd.read_csv(param.testdev)
df_test_dev = dsm.remove_single_clickout_actions(df_test_dev)
df_test_dev = dsm.remove_nonitem_actions(df_test_dev)
'''

'''
logfile.write('Start dev submission - Time: ' + str(timeSince(start_test_time)) + '\n')

mrr = tst.test_accuracy_optimized_classification(model, df_test_dev, df_gt_inner, test_session_dict, test_category_dict, test_impression_dict, hotel_dict, hotel_to_index_dict, n_features, max_window, meta_dict, meta_list, dir, param.subname, isprint=True, dev = True)
#print("Final score for dev: " + str(mrr))
print(timeSince(start_test_time))

logfile.write('Finish dev submission - Time: ' + str(timeSince(start_test_time)) + '\n')
'''

'''
STEP 8: SAVING SUBMISSION
'''

#Computing score
#print("End execution with score " + str(mrr))
'''
file_exists = os.path.isfile('classification_scores.csv')
with open('classification_scores.csv', mode='a') as score_file:
    file_writer = csv.writer(score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    if not file_exists: # Write headers
        file_writer.writerow(['Train set', 'Using impressions', 'Using meta', 'Hidden dimension', 'Dropout layer', '#Epochs', '#Components', 'W2Vec window', 'Learn Rate', 'batchsize', 'Score'])
    #file_writer.writerow([str(param.train), str(param.isimpression), str(param.ismeta), str(param.hiddendim), str(param.isdrop), str(param.epochs), str(param.ncomponents), str(param.window), str(param.learnrate), str(param.batchsize), str(mrr)])
#f.send_telegram_message("End execution with score " + str(mrr))
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