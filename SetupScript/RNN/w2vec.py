import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

def normalize_word2vec(word2vec):
  hotels_pre_norm = []

  for hotel in word2vec.wv.index2word:
    hotels_pre_norm.append(word2vec.wv[hotel].tolist())

  hotels_pre_norm = np.asarray(hotels_pre_norm)
  hotels_post_norm = normalize(hotels_pre_norm, norm='l2', axis=0, copy=True, return_norm=False)
  
  hotels_post_norm = hotels_post_norm.tolist()

  for hotel in word2vec.wv.index2word:
    word2vec.wv[hotel] = np.asarray(hotels_post_norm[0])
    hotels_post_norm.pop(0)
    
  return word2vec