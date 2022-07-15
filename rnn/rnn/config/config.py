import numpy as np
import io
import torch
from torch import nn
import torch.nn.functional as F

class Config:
    '''
    Config class defines dataset path and hyperparameters.
    '''
    data_train_url = 'rnn/rnn/data/shakespeare_train.txt'
    data_val_url = 'rnn/rnn/data/shakespeare_valid.txt'
    model_path = 'rnn/rnn/models/rnn.net'
    n_hidden = 512
    n_layers = 2
    epochs = 25 
    n_seqs = 128
    n_steps = 100
    lr = 0.001
    clip = 5
    cuda = False
    dropout = 0.5