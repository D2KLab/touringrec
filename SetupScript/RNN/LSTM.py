import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, iscuda, bias=True):
        super(LSTMModel, self).__init__()

        # Cuda flag
        self.iscuda = iscuda

        # Hidden dimensions
        self.hidden_dim = hidden_dim
         
        # Number of hidden layers
        self.layer_dim = layer_dim
               
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = layer_dim)  
        
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim * 2)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    
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

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.hidden_fc(out)

        out = out[-1, :, :]
        
        out = self.fc(out)
    
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

def session_to_tensor(session, hotel_dict, n_features):
  tensor = torch.zeros(len(session), 1, n_features)
  
  for ai, action in enumerate(session):
    tensor[ai][0] = hotel_to_tensor(action['reference'], hotel_dict, n_features)
  return tensor

def sessions_to_batch(session_list, hotel_dict, max_session_len, n_features):
  batch_dim  = len(session_list)

  tensor = torch.zeros(max_session_len, batch_dim, n_features)
  
  for si, session in enumerate(session_list):
    for ai, action in enumerate(session):
      tensor[ai][si] = hotel_to_tensor(action['reference'], hotel_dict, n_features)
  return tensor


def hotel_to_tensor(hotel, hotel_dict, n_features):
  tensor = torch.zeros(n_features)
  if hotel in hotel_dict: #-----------int
    tensor = torch.from_numpy(hotel_dict[hotel])
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

def category_from_output(output, hotel_dict):
  top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
  category_i = int(top_i[0][0])
  #print(output)
  return hotel_dict.index2word[category_i], category_i
  
