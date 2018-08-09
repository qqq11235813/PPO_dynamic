import numpy as np
import sys
import codecs
import os
import math
import operator
import json
from functools import reduce
from random import shuffle
import matplotlib.pyplot as plt
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


def words2sentence(words):
    output = ''
    for word in words:
        output += ' '
        output += word
    return output

def fetch_data(cand, ref):
    """ Store each reference and candidate sentences as a list """
    references = []
    if '.txt' in ref:
        reference_file = codecs.open(ref, 'r', 'utf-8')
        references.append(reference_file.readlines())
    else:
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference_file = codecs.open(os.path.join(root, f), 'r', 'utf-8')
                references.append(reference_file.readlines())
    candidate_file = codecs.open(cand, 'r', 'utf-8')
    candidate = candidate_file.readlines()
    return candidate, references


def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(candidate, references, n = 2):
    precisions = []
    for i in range(n):
        pr, bp = count_ngram(candidate, references, i+1)
        precisions.append(pr)
    bleu = geometric_mean(precisions) * bp
    return bleu
  

def v2c(x):
    if use_cuda:
        return x.cuda()
    else:
        return x

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def generate_save_text(encoder, decoder, evaluater, eva_encoder, eva_decoder):
    text_e = 'encoder:hs:' + str(encoder.hidden_size)
    text_d = 'decoder:hs' + str(decoder.hidden_size)
    text_evaluater = 'evaluater' 
    text_eva_e = 'eva_encoder'
    text_eva_d = 'eva_decoder'
    return text_e, text_d, text_evaluater, text_eva_e, text_eva_d


SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3

class Lang_counting:
    def __init__(self, name):
        self.name = name
        self.word2index = {"100":0, "101":1, "103":3, "102":2}
        self.word2count = {}
        self.index2word = {0: "100", 1: "101", 3:"103", 2:"102"}
        self.n_words = 14  # Count SOS and EOS
        
        for i in range(10):
            self.word2index[str(i)] = i+4
            self.index2word[i+4] = str(i)

class Lang_real:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<SOS>":0, "<EOS>":1, "<UNK>":2, "<PAD>":3}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2:"<UNK>", 3:"<PAD>"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        #sentence = sentence.replace('.', '')
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

            
def filterPairs(pairs):
    return [pair for pair in pairs]


def indexesFromSentence(lang, sentence, max_length):
    #sentence = sentence.replace('.', '')
    index = []
    for word in sentence.split(' '):
        if word in lang.index2word.values():
            index.append(lang.word2index[word])
        else:
            index.append(UNK_token)
      
    len_indexes = len(index)
  
    index.append(EOS_token)
  
    for _ in range(max_length - len_indexes + 2):
        index.append(PAD_token)  
      
    return index

def variableFromSentence(lang, sentence, max_length):
    indexes = indexesFromSentence(lang, sentence, max_length)
    
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result
      
def sens2tensor(sens, lang):
    max_len = 0
    for sen in sens:
        max_len = max(max_len, len(sen.split(' ')))
        
    sen_indexes = []
    for sen in sens:
        sen_indexes.append(indexesFromSentence(lang, sen, max_len))
        
    sen_indexes = np.array(sen_indexes)
    tensor = []
    for n in range(max_len + 1):
        sen = Variable(torch.LongTensor(sen_indexes[:, n].reshape((1, -1))))
        sen = v2c(sen)
        tensor.append(sen)
    
    return tensor

def get_dataset(file_dict, batch_size = 8, lang_txt = Lang_real('train_txt'), task = 'counting'):
    
    '''
    split the whole dataset into each batch and change the type of it
    Used for generating training_set and dev_set
    input:
        file_dict: the direction of the txt file
        batch_size: parameter
        lang_txt: the Lang type class.changed by the task
    task: different data type.
        Two types: counting, real
        
    output:
        output_data: the list of the data. Each element is one batch
            type: list
            element: one batch of data.  
            (train_sens_tensor, label_sens_tensor, train_sens)
                type: turple
                element: 
                    train_sens_tensor: the tensor(input sentence) of the input of the encoder
                        type: list
                        list_length: the input sentence length
                        element: one word of a batch(tensor)
                            type: pytorch variable
                            shape: (1, batch_size)
                    label_sens_tensor: the tensor(target sentence) of the input of the decoder
                        type: list
                        list_length: the target sentence length
                        element: one word of a batch(tensor)
                            type: pytorch variable
                            shape: (1, batch_size)
                    
                    train_sens: the list of one batch of sentence(used for testing bleu score and accuracy)
                        type: list
                        list_length: batch_size
                        element: target sentence
                            type: string
        lang_txt: lang class
    '''
    
    #lang_txt = Lang('train_txt')
    lines = open(file_dict).read().strip().split('\n')
   
    len_lines = len(lines)
    
    dataset = []
    
    for index, sen in enumerate(lines):
        if(index % 2 == 0):
            next_sen = lines[index + 1]
            pair = (sen, next_sen)
            max_len = max(len(sen.split(' ')), len((next_sen.split(' '))))
            dataset.append((max_len, pair))
    dataset = sorted(dataset)
    
    print('dataset prepared')
    
    N = len(dataset)
    batch_num = int(np.ceil(N/batch_size))
    
    batch_data = [None] * batch_num 
    for n in range(batch_num):
        batch_data[n] = ([],[])
        
    for index, ele in enumerate(dataset):
        
        #if index % 20000 == 0:
            #print('completed', index/(N+ 0.0))
        
        _, pair = ele
        train_sen , label_sen = pair
        batch_index = int(index/batch_size)
        sens, labels = batch_data[batch_index]
        sens.append(train_sen)
        labels.append(label_sen)
    
    print('split prepared')
    
    output_data = []
    for index, ele in enumerate(batch_data):
        #if index % 200 == 0:
            #print('finished', index/(batch_num + 0.0))
        train_sens, label_sens = ele
        #set_trace()
        train_sens_tensor = sens2tensor(train_sens, lang_txt)
        label_sens_tensor = sens2tensor(label_sens, lang_txt)
        
        if(task == 'counting'):
            output_data.append((train_sens_tensor, label_sens_tensor, train_sens))
        elif(task == 'real'):
            output_data.append((train_sens_tensor, label_sens_tensor, label_sens))
    return output_data



def get_dataset_test_counting(file_dict, batch_size = 8, lang_txt = Lang_counting('train_txt')):
    '''
    split the whole dataset into each batch and change the type of it
    Used for generating test_set
    input:
        file_dict: the direction of the txt file
        batch_size: parameter
        lang_txt: the Lang type class.changed by the task
   
    output:
        output_data: the list of the data. Each element is one batch
            type: list
            element: one batch of data.  
            (train_sens_tensor, label_sens_tensor, train_sens)
                type: turple
                element: 
                    train_sens_tensor: the tensor(input sentence) of the input of the encoder
                        type: list
                        list_length: the input sentence length
                        element: one word of a batch(tensor)
                            type: pytorch variable
                            shape: (1, batch_size)
                    label_sens_tensor: the tensor(target sentence) of the input of the decoder
                        type: list
                        list_length: the target sentence length
                        element: one word of a batch(tensor)
                            type: pytorch variable
                            shape: (1, batch_size)
                    
                    train_sens: the list of one batch of sentence(used for testing bleu score and accuracy)
                        type: list
                        list_length: batch_size
                        element: target sentence
                            type: string
        lang_txt: lang class
    '''
    
    #lang_txt = Lang('train_txt')
    lines = open(file_dict).read().strip().split('\n')
   
    len_lines = len(lines)
    
    dataset = []
    
    for index, sen in enumerate(lines):
        next_sen = '0 0 0'
        pair = (sen, next_sen)
        max_len = max(len(sen.split(' ')), len((next_sen.split(' '))))
        dataset.append((max_len, pair))
    dataset = sorted(dataset)
    
    print('dataset prepared')
    
    N = len(dataset)
    batch_num = int(np.ceil(N/batch_size))
    
    batch_data = [None] * batch_num 
    for n in range(batch_num):
        batch_data[n] = ([],[])
        
    for index, ele in enumerate(dataset):
        
       # if index % 20000 == 0:
           # print('completed', index/(N+ 0.0))
        
        _, pair = ele
        train_sen , label_sen = pair
        batch_index = int(index/batch_size)
        sens, labels = batch_data[batch_index]
        sens.append(train_sen)
        labels.append(label_sen)
    
    print('split prepared')
    
    output_data = []
    for index, ele in enumerate(batch_data):
        #if index % 200 == 0:
            #print('finished', index/(batch_num + 0.0))
        train_sens, label_sens = ele
        #set_trace()
        train_sens_tensor = sens2tensor(train_sens, lang_txt)
        label_sens_tensor = sens2tensor(label_sens, lang_txt)
        output_data.append((train_sens_tensor, label_sens_tensor, train_sens))
            
    return output_data



