import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
import numpy as np
import pandas as pd
import csv
import ds_manipulation as dsm
import w2vec as w2v
import test_f as tst
import LSTM as lstm
import LSTMParameters as LSTMParam


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start_program_time = time.time()


df_train_inner = pd.read_csv('./train_off.csv') ###
df_train_inner = dsm.remove_single_clickout_actions(df_train_inner)
df_train_inner =  dsm.remove_nonitem_actions(df_train_inner)
df_train_inner = dsm.reference_to_str(df_train_inner)
session_dict = {}
category_dict = {}
impression_dict = {}
session_dict, category_dict, impression_dict, train_corpus = dsm.get_training_input(df_train_inner)
del session_dict
del category_dict
del df_train_inner


df_test_inner = pd.read_csv('test_off.csv') ###
df_test_inner = dsm.remove_single_clickout_actions(df_test_inner)
df_test_inner = dsm.remove_nonitem_actions(df_test_inner)
df_test_for_prepare = dsm.reference_to_str(df_test_inner.copy())
test_session_dict = {}
test_category_dict = {}
test_impression_dict = {}
test_session_dict, test_category_dict, test_impression_dict, test_corpus = dsm.get_test_input(df_test_for_prepare)
del df_test_for_prepare


'''test_dev_session_dict = {}
test_dev_category_dict = {}
test_dev_impression_dict = {}
test_dev_session_dict, test_dev_category_dict, test_dev_impression_dict, test_dev_corpus = dsm.get_test_input(df_test_dev_for_prepare)
df_test_dev = pd.read_csv(param.testdev) ###
df_test_dev = dsm.remove_single_clickout_actions(df_test_dev)
df_test_dev = dsm.remove_nonitem_actions(df_test_dev)
df_test_dev_for_prepare = dsm.reference_to_str(df_test_dev.copy())'''
#del...


corpus = train_corpus + test_corpus

#w2vec item encoding
from gensim.models import Word2Vec

word2vec = Word2Vec(corpus, size = 60, min_count=1, window=5, sg=1)
#del train_corpus
del test_corpus

n_features = len(word2vec.wv['666856'])

hotel_dict = {k: list(word2vec.wv[k]) for k in word2vec.wv.index2word}

del word2vec
del corpus

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
#hidden_dim = int(1/100 * (input_dim + output_dim))
hidden_dim = 100
print('The model is:')
print('input_dim is:' + str(input_dim))
print('hidden_dim is: ' + str(hidden_dim))
print('output_dim is:' + str(output_dim))
layer_dim = 1

#NET CREATION
#model = lstm.LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, True)
#model.load_state_dict(torch.load('./ultimate/model_epoch_50_10%')) ####
#model.eval()

#model = model.cuda()

#LOSS FUNCTION
#loss_fn = torch.nn.CrossEntropyLoss()
#loss_fn = torch.nn.NLLLoss()


#loss_fn = loss_fn.cuda()

#OPTIMIZER
#optimizer = torch.optim.Adam(model.parameters(), lr=param.learnrate)
#model.optimizer = optimizer

def list_to_padded(t):
    k = t[0]
    v = t[1]
    if k in test_category_dict:
        l = list(v)
        l = list(map(lambda h: hotel_dict[h], l))
        miss_n = 200 - len(l)
        for missing in range(miss_n):
            l.append([0] * 60)
        return l
    else:
        return []

test_corpus = list(map(lambda t: list_to_padded(t), test_session_dict.items()))
test_corpus = [x for x in test_corpus if x != []]
tensor_input = torch.FloatTensor(test_corpus)


dataloader = DataLoader(dataset = tensor_input, batch_size = 1)

print(timeSince(start_program_time))