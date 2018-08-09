from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3


class EncoderRNN(nn.Module):
    def __init__(self, feature_size, hidden_size, output_length, layer = 1):
        super(EncoderRNN, self).__init__()
        ''' 
        init parameters:
          feature_size: the feature number of the embedding layer
          hidden_size: the hidden size of GRU(usually equal to feature size)
          output_length: the number of total dictionary
          layer: layer number of RNN
        '''
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_length, feature_size, padding_idx = PAD_token)
        self.gru = nn.GRU(feature_size, hidden_size, num_layers = layer)
        self.gru.flatten_parameters()
        self.layer = layer
        
    def forward(self, input, hidden):
        '''
        input:
            input: one batch of word(changed by dictionary)
                type: pytorch variable LongTensor
                size: (1, batch_size)
            hidden: hidden state of the RNN
                type: pytorch variable FloatTensor
                size: (layer_number, batch_size, hidden_size)
        output:
            output: output of RNN unit
                type: pytorch variable FloatTensor
                size: (layer_number, batch_size, feature_size)
            hidden: hidden state of RNN unit
                type: pytorch variable FloatTensor
                size: (layer_number, batch_size, feature_size)
        '''
        embedded = self.embedding(input.view(1, -1))
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.layer, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
   
  
MAX_LENGTH = 150

class DecoderRNN(nn.Module):
    
    def __init__(self, feature_size, hidden_size, output_length, layer = 1):
        ''' 
        init parameters:
          feature_size: the feature number of the embedding layer
          hidden_size: the hidden size of GRU(usually equal to feature size)
          output_length: the number of total dictionary
          layer: layer number of RNN
        '''
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_length = output_length
        self.feature_size = feature_size
        
        self.embedding = nn.Embedding(output_length, feature_size, padding_idx = PAD_token)
        self.gru = nn.GRU(feature_size, hidden_size, num_layers = layer)
        self.gru.flatten_parameters()

        self.out = nn.Linear(feature_size, output_length)
        self.softmax = nn.LogSoftmax(dim=2)
        
        self.layer = layer
        
    def forward(self, input, hidden, rubbish):
        '''
        input:
            input: one batch of word(changed by dictionary)
                type: pytorch variable LongTensor
                size: (1, batch_size)
            hidden: hidden state of the RNN
                type: pytorch variable FloatTensor
                size: (layer_number, batch_size, hidden_size)
        output:
            output: output of RNN unit
                type: pytorch variable FloatTensor
                size: (layer_number, batch_size, feature_size)
            hidden: hidden state of RNN unit
                type: pytorch variable FloatTensor
                size: (layer_number, batch_size, feature_size)
        '''
        output = self.embedding(input.view(1, -1))
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output))
        return output, hidden, None
      
    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.layer, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result 
        

class EvaluateDecoder(nn.Module):
    def __init__(self, feature_size, hidden_size, output_length):
        super(EvaluateDecoder, self).__init__()
        '''
        feature_size: input of GRU layer dimension
        hidden_size: hidden size of GRU 
        output_length:the length of wrod
        
        e.g. 
        EvaluateCritic(256, 256, lang_txt.n_words)
        '''
        self.hidden_size = hidden_size
        self.output_length = output_length
        self.feature_size = feature_size
        
        self.embedding = nn.Embedding(output_length, feature_size, padding_idx = PAD_token)
        self.gru = nn.GRU(feature_size, hidden_size)
        self.gru.flatten_parameters()

        self.out = nn.Linear(hidden_size, output_length)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input, hidden, rubbish):
        '''
        input: a batch of a word index.
                  should be LongTensor, shape = (1, batch_size)
        
        '''
        output = self.embedding(input.view(1, -1))
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.sigmoid(self.out(output))
        return output, hidden, None
      
    def initHidden(self, batch_size):
        '''
        return the hidden of the first GRU cell
        batch_size:int
        '''
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result  
        
        
class EvaluateEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, output_length):
        super(EvaluateEncoder, self).__init__()
        '''
        feature_size: input of GRU layer dimension
        hidden_size: hidden size of GRU 
        output_length:the length of wrod
        
        e.g. 
        EvaluateCritic(256, 256, lang_txt.n_words)
        '''
        self.hidden_size = hidden_size
        self.output_length = output_length
        self.feature_size = feature_size
        
        self.embedding = nn.Embedding(output_length, feature_size, padding_idx = PAD_token)
        self.gru = nn.GRU(feature_size, hidden_size)
        self.gru.flatten_parameters()

    def forward(self, input, hidden):
        '''
        input: a batch of a word index.
                  should be LongTensor, shape = (1, batch_size)
        hidden: hidden of GRU
        
        '''
        output = self.embedding(input.view(1, -1))
        output, hidden = self.gru(output, hidden)
        return output, hidden
      
    def initHidden(self, batch_size):
        '''
        return the hidden of the first GRU cell
        batch_size:int
        '''
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result  
        

class EvaluateR(nn.Module):
    '''
    the evaluation function
    use one layer of nn
    '''
    def __init__(self, hidden_size, layer_num = 1):
        super(EvaluateR, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.linear = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        '''
        input:the first layer of hidden state
            type: pytorch FloatTensor variable 
            shape: (batch_size, feature_number)
        output: the evauate score of each word
            type: pytorch FloatTensor variable(from 0 to 1)
            shape: (batch_size, 1)
        '''
        #input1 = input.view(1, -1)
        input = input.view(-1, self.hidden_size)
        output = self.sigmoid(self.linear(input))
        return output

class disEncoderRNN(nn.Module):
    def __init__(self, feature_size, hidden_size, output_length):
        super(disEncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_length, feature_size)
        self.gru = nn.GRU(feature_size, hidden_size)
        #self.gru.flatten_parameters()
        
    def forward(self, input, hidden):
        embedded = self.embedding(input.view(1, -1))
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda(0)
        else:
            return result
          
class disDecoderRNN(nn.Module):
    def __init__(self, feature_size, hidden_size, output_length, dropout = 0.2):
        super(disDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_length = output_length
        self.feature_size = feature_size
        
        self.embedding = nn.Embedding(output_length, feature_size)
        self.gru = nn.GRU(feature_size, hidden_size)
        self.gru.flatten_parameters()
        
        self.dropout_linear = nn.Dropout(p = dropout)
        self.out = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input, hidden, rubbish):
        
        output = self.embedding(input.view(1, -1))
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.dropout_linear(output)
        output = self.sigmoid(self.out(output))
        return output, hidden, None
      
    def initHidden(self, batch_size):
        
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda(0)
        else:
            return result  
