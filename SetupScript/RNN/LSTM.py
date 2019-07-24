import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from operator import itemgetter
import operator
import time
import math
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

        self.gru = nn.GRU(input_size = input_dim, hidden_size = hidden_dim, num_layers = layer_dim) # Use dropout with more than 1 layer

        self.fc = nn.Linear(hidden_dim, output_dim)
    
        self.softmax = nn.Softmax(dim = 1)

        self.logsoftmax = nn.LogSoftmax(dim = 1)
    
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

        out, hn = self.gru(x, h0.detach())

        out = out[-1, :, :]

        out = self.softmax(out)
        
        out = self.fc(out)
    
        out = self.logsoftmax(out)

        return out

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(model, loss_fn, optimizer, category_tensor, line_tensor, iscuda):
    
    optimizer.zero_grad()
    
    line_tensor = line_tensor.requires_grad_()

    start = time.time()

    output = model(line_tensor)

    if iscuda:
      category_tensor = category_tensor.long().cuda()

    loss = loss_fn(output, category_tensor)
    loss.backward()

    optimizer.step()
    
    return output, loss.item()


#functions for training phase

def session_to_tensor_ultimate(session, hotel_dict, n_features, hotels_window):
  tensor = torch.zeros(len(session), 1, n_features)
  
  for hi, hotel in enumerate(session):
    tensor[hi][0] = hotel_dict[hotel]

  return tensor

def sessions_to_batch_tensor(session_list, session_dict, hotel_dict, max_session_len, n_features):
  batch_dim  = len(session_list)

  tensor = torch.zeros(max_session_len, batch_dim, n_features)

  for si, session in enumerate(session_list):
    for hi, hotel in enumerate(session_dict[session]):
      tensor[hi][si] = hotel_dict[hotel]
      
  return tensor

def meta_to_index(meta, meta_list):
    return meta_list.index(meta)

def hotels_to_category_batch(hotel_list, hotel_to_category_dict, n_hotels):
  batch_size = len(hotel_list)
  tensor = torch.zeros(batch_size)
  
  for hi, hotel in enumerate(hotel_list):
    if hotel in hotel_to_category_dict:
      tensor[hi] = hotel_to_category_dict[hotel]

  return tensor

def assign_score(hotel, output_arr, hotel_to_index_dict):
  if hotel not in hotel_to_index_dict:
    return (hotel, -999)
  else:
    return (hotel, output_arr[hotel_to_index_dict[hotel]])

def categories_from_output_windowed_opt(output, batch, impression_dict, hotel_dict, hotel_to_index_dict, df_train_inner_sub_list, pickfirst = False):
  output_arr = np.asarray(output.cpu().detach().numpy())
  
  categories_batched = []
  categories_scores_batched = []

  for batch_i, single_session in enumerate(batch):
    window = impression_dict[single_session]

    category_tuples = list(map(lambda x: assign_score(x, output_arr[batch_i], hotel_to_index_dict), window))
    category_tuples = sorted(category_tuples, key=lambda tup: tup[1], reverse = True)

    # Converting to 2 lists
    category_dlist = list(map(list, zip(*category_tuples)))

    categories_batched.append(category_dlist[0])
    categories_scores_batched.append(category_dlist[1])

  return categories_batched, categories_scores_batched