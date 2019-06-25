import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from operator import itemgetter
#from setup import param

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, iscuda, bias=True):
        super(LSTMModel, self).__init__()

        # Cuda flag
        self.iscuda = iscuda

        # Hidden dimensions
        self.hidden_dim = hidden_dim
         
        # Number of hidden layers
        self.layer_dim = layer_dim

        # Layers

        #self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = layer_dim)  
      
        self.gru = nn.GRU(input_size = input_dim, hidden_size = hidden_dim, num_layers = layer_dim) # Use dropout with more than 1 layer

        #self.hidden_fc = nn.Linear(hidden_dim, hidden_dim * 10)

        self.fc = nn.Linear(hidden_dim, output_dim)
    
        self.softmax = nn.Softmax(dim = 1)

        self.logsoftmax = nn.LogSoftmax(dim = 1)

        #self.dropout_layer = nn.Dropout(p=0.2)
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        if self.iscuda:
            h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim)

        # Initialize cell state
        if self.iscuda:
            c0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim).cuda()
        else:
            c0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim)

        # Tensor to cuda if necessary
        if self.iscuda:
          x = x.cuda()

        #out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out, hn = self.gru(x, h0.detach())

        #out = F.relu(self.hidden_fc(out))

        out = out[-1, :, :]

        out = self.softmax(out)

        #out = self.dropout_layer(out)
        
        out = self.fc(out)
    
        out = self.logsoftmax(out)

        return out


def train(model, loss_fn, optimizer, category_tensor, line_tensor, iscuda):
    
    optimizer.zero_grad()
    
    line_tensor = line_tensor.requires_grad_()

    output = model(line_tensor)
    
    if iscuda:
      category_tensor = category_tensor.long().cuda()

    loss = loss_fn(output, category_tensor)
    loss.backward()

    optimizer.step()
    
    return output, loss.item()


#functions for training phase

def session_to_tensor(session, hotel_dict, n_features, hotels_window, max_window, meta_dict, meta_list):
  tensor = torch.zeros(len(session), 1, n_features)
  
  for ai, action in enumerate(session):
    tensor[ai][0] = hotel_to_tensor(action['reference'], hotel_dict, n_features, hotels_window, max_window, meta_dict, meta_list)
  return tensor

def sessions_to_batch(session_list, hotel_dict, max_session_len, n_features, hotels_window, max_window, meta_dict, meta_list):
  batch_dim  = len(session_list)

  tensor = torch.zeros(max_session_len, batch_dim, n_features)
  
  for si, session in enumerate(session_list):
    for ai, action in enumerate(session):
      tensor[ai][si] = hotel_to_tensor(action['reference'], hotel_dict, n_features, hotels_window, max_window, meta_dict, meta_list)
  return tensor

def meta_to_index(meta, meta_list):
    return meta_list.index(meta)

def hotel_to_tensor(hotel, hotel_dict, n_features, hotels_window, max_window, meta_dict, meta_list):
  n_features_w2vec = 100 #to be fixed
  n_features_meta = len(meta_list)
  tensor_w2vec = torch.zeros(n_features_w2vec)
  tensor_meta = torch.zeros(n_features_meta)
  tensor_window = torch.zeros(max_window)
  
  if hotel in hotel_dict:
    tensor_w2vec = torch.from_numpy(hotel_dict[hotel])
  
  if hotel in meta_dict:
    for mi, meta in enumerate(meta_dict[hotel]):
      tensor_meta[meta_to_index(meta, meta_list)] = 1
    
  if max_window != 0:  
    if hotel in hotels_window:
      tensor_window[hotels_window.index(hotel)] = 1
      
  tensor = torch.cat((tensor_w2vec, tensor_meta), 0)
  tensor = torch.cat((tensor, tensor_window), 0)
    
  return tensor

def hotel_to_category(hotel, hotel_dict, n_features):
  tensor = torch.zeros(1)
  if hotel in hotel_dict.index2word:
    tensor = torch.tensor([hotel_dict.index2word.index(hotel)], dtype=torch.long)
  return tensor

def hotels_to_category_batch(hotel_list, hotel_dict, n_hotels):
  batch_size = len(hotel_list)
  tensor = torch.zeros(batch_size)
  for hi, hotel in enumerate(hotel_list):
    if hotel in hotel_dict.index2word:
      tensor[hi] = torch.tensor([hotel_dict.index2word.index(hotel)], dtype=torch.long)
  return tensor

'''def category_from_output(output, hotel_dict):
  #top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
  #category_i = int(top_i[0][0])
  
  category_score, category_i = torch.max(output, 1)
  
  #print(output)
  return hotel_dict.index2word[category_i], category_i'''
  

def category_from_output(output, hotel_dict):
  #top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
  #category_i = int(top_i[0][0])
  
  
  category_score, category_i = torch.max(output, 1)
  
  categories = []
  
  #print(output)
  #print(category_score)
  #print(category_i)
  
  for cat_i, cat in enumerate(category_i):
    categories.append(hotel_dict.index2word[cat])
  
  #print(output)
  return categories, category_i

def categories_from_output_windowed_opt(output, hotel_window, hotel_dict, pickfirst = False):
  output_arr = np.asarray(output.cpu().detach().numpy())
  
  category_scores_dict = {}
  categories_scores = []
  categories = []

  for batch_i, window in enumerate(hotel_window):
    category_scores_dict = {}
    for hotelw_i, hotelw in enumerate(window):
      if hotelw in hotel_dict:
        hotel_i = hotel_dict.index2word.index(hotelw)
        category_scores_dict[hotelw] = output_arr[batch_i][hotel_i]
      else:
        category_scores_dict[hotelw] = -9999
        
    #print(category_scores_dict)
    category_scores_tuples = sorted(category_scores_dict.items(), key=itemgetter(1), reverse = True)
    #print(category_scores_tuples)
    temp_categories = []
    temp_scores = []
      
    if pickfirst:
      temp_categories = category_scores_tuples[0][0]
      temp_scores = category_scores_tuples[0][1]
      
    else:  
      for tup in category_scores_tuples:
        temp_categories.append(tup[0])
        temp_scores.append(tup[1])
      
    categories.append(temp_categories)
    categories_scores.append(temp_scores)
  
  return categories, categories_scores