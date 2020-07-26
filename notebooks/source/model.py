import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, init_weights='none', dropout_rate=0.5):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(Net, self).__init__()
        
        print("Making Net with input {}, hidden {}".format(input_dim, hidden_dim))

        self.model = nn.Sequential(
            
            # Input layer
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),

            # Hidden layers
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
                       
            # Output layer
            nn.Linear(hidden_dim, 1),
        )
        self.sig = nn.Sigmoid()

        if init_weights == 'xavier_uniform':
            self.model.apply(Net.init_weights_xavier_uniform)
        elif init_weights == 'xavier_normal':
            self.model.apply(Net.init_weights_xavier_normal)
        elif init_weights == 'uniform':
            self.model.apply(Net.init_weights_uniform)

    @staticmethod
    def init_weights_xavier_uniform(model):
        if type(model) == nn.Linear:
            print("Setting initial weights on {}".format(str(model)))
            torch.nn.init.xavier_uniform_(model.weight)
            if model.bias is not None:
                model.bias.data.fill_(0)    

    @staticmethod
    def init_weights_xavier_normal(model):
        if type(model) == nn.Linear:
            print("Setting initial weights on {}".format(str(model)))
            torch.nn.init.xavier_normal_(model.weight)
            if model.bias is not None:
                model.bias.data.fill_(0)    

    @staticmethod
    def init_weights_uniform(model):
        if type(model) == nn.Linear:
            print("Setting initial weights on {}".format(str(model)))
            n = model.in_features
            y = 1.0/np.sqrt(n)
            print("Y={}".format(y))
            model.weight.data.uniform_(-y, y)
            if model.bias is not None:
                model.bias.data.fill_(0)    


    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''

        x = self.model(x)
        return self.sig(x)
