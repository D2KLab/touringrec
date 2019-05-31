import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim
         
        # Number of hidden layers
        self.layer_dim = layer_dim
               
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = layer_dim)  
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim)

        # Initialize cell state
        if torch.cuda.is_available():
            c0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim).cuda()
        else:
            c0 = torch.zeros(self.layer_dim, x.size(1), hidden_dim)

            
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        
        out = out[-1, :, :]
        
        out = self.fc(out)
    
        return out


def train(category_tensor, line_tensor):
    hidden = torch.zeros(1, 1, n_hidden)
    c = torch.zeros(1, 1, n_hidden)
    
    optimizer.zero_grad()
    
    line_tensor = line_tensor.requires_grad_()
    line_tensor = line_tensor.cuda()
    

    output = model(line_tensor)
    
    category_tensor = category_tensor.long().cuda()

    loss = loss_fn(output, category_tensor)
    loss.backward()

    optimizer.step()
    
    return output, loss.item()


#functions for training phase

def session_to_tensor(session):
  tensor = torch.zeros(len(session), 1, n_features)
  
  for ai, action in enumerate(session):
    tensor[ai][0] = hotel_to_tensor(action['reference'], hotel_dict, n_features)
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

def category_from_output(output):
  top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
  category_i = int(top_i[0][0])
  #print(output)
  return hotel_dict.index2word[category_i], category_i
  
