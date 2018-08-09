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
from prepare import *

MAX_LENGTH = 150

class seq2seq(nn.Module):
  
    def __init__(self,lang_txt,
               dev_data, #dev set 
               test_data, #test set
               encoder,
               decoder,
               evaluater,
               #eva_encoder,
               #eva_decoder,
               #de_encoder, 
               #de_decoder,
               #de_eva_encoder,
               #de_eva_decoder,
               encoder_prev, #the encoder to record the previous state
               decoder_prev,
                 
               task = 'counting',
               max_length = 150,
               god_rs = [], #the data recorded
               god_rs_dev = [],
               god_loss_dev = [],
               god_loss = [],
               god_rs_test = [],
               decay_rate = 1.0
    ):
        super(seq2seq,self).__init__()
        self.lang_txt = lang_txt
        self.dev_data = dev_data
        self.test_data = test_data
        self.task = task
        
        self.encoder = encoder
        self.decoder = decoder
        self.evaluater = evaluater
        #self.eva_encoder = eva_encoder
        #self.eva_decoder = eva_decoder

        #self.de_encoder = de_encoder
        #self.de_decoder = de_decoder
        #self.de_eva_encoder = de_eva_encoder
        #self.de_eva_decoder = de_eva_decoder

        self.decoder_prev = decoder_prev
        self.encoder_prev = encoder_prev

        self.learning_rate = 0.0001

        self.hidden_size = encoder.hidden_size

        self.criterion = nn.NLLLoss()
        self.evaluater_criterion = nn.MSELoss()


        # self.decay_rate = Variable(torch.FloatTensor([decay_rate]))
        #self.decay_rate = self.decay_rate.cuda() if use_cuda else self.decay_rate

        self.bleu = v2c(Variable(torch.FloatTensor([0.5])))
        self.ave_bleu = v2c(Variable(torch.FloatTensor([0.5])))

        self.sentence_step = 0  #the REINFORCE step length(max:target_length, min:-delta)
        self.sentence_delta = 3 #the REINFORCE delta

        self.max_sentence_length = 0

        self.flag_god = False

        self.god_rs = god_rs
        self.god_rs_dev = god_rs_dev
        self.god_loss_dev = god_loss_dev
        self.god_loss = god_loss
        self.god_rs_test = god_rs_test
        self.lr = 0.001

    def dev_score(self, dev_data, types = 1):
        '''
        get the score of the model.
        input: dev set / test set data
          type: data_type
        output: (average reward, average losses)
        '''
        
        if(types == 1):
            #dev_data = self.dev_data
            f_rs = []
            losses = []
            encoder = self.encoder
            decoder = self.decoder

            for index,batch in enumerate(dev_data):
                input_variable, target_variable, target_sentence = batch

                batch_size = input_variable[0].size()[1]
                input_length = len(input_variable)
                target_length = len(target_variable)
                batch_sentence = [None] * batch_size
                for i in range(batch_size):
                    batch_sentence[i]  = ''

                encoder_hidden = encoder.initHidden(batch_size)
                EOS_flag_list = [None] * batch_size
                for i in range(batch_size):
                    EOS_flag_list[i] = False

                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(
                        input_variable[ei], encoder_hidden)

                start_token = np.ones((1, batch_size)) * SOS_token
                decoder_input = Variable(torch.LongTensor(start_token))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                decoder_hidden = encoder_hidden

                loss = 0

                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, None)

                    loss = self.criterion(decoder_output[0], target_variable[di][0])
                    losses.append(loss.data[0])

                    topv, topi = decoder_output.data.topk(1)
                    ni = topi.view(1,-1)
                    decoder_input = v2c(Variable(ni))
                    for i in range(batch_size):

                        if EOS_flag_list[i]:
                            decoder_input[0][i].data[0] = PAD_token
                            word_index = PAD_token
                        else: 
                            word_index = ni[0][i].item()


                        if(word_index == EOS_token) or (word_index == PAD_token):
                            EOS_flag_list[i] = True
                            continue
                        decoder_word = self.lang_txt.index2word[word_index]

                        if(batch_sentence[i] == ''):
                            batch_sentence[i] += (decoder_word)
                        else:
                            batch_sentence[i] += (' ' + decoder_word)
                rewards = self.get_reward(batch_sentence, target_sentence, EOS_flag_list, types = types)
                reward = np.mean(rewards)
                f_rs.append(reward)
                if(self.flag_god):
                    print(rewards)
            f_r = np.mean(f_rs)

            return f_r, np.mean(losses)
        
        elif(types == 2):
            #dev_data = self.dev_data
            f_rs = []
            losses = []
            encoder = self.encoder
            decoder = self.decoder

            for index,batch in enumerate(dev_data):
                input_variable, target_variable, target_sentence = batch

                batch_size = input_variable[0].size()[1]
                input_length = len(input_variable)
                target_length = MAX_LENGTH
                batch_sentence = [None] * batch_size
                for i in range(batch_size):
                    batch_sentence[i]  = ''

                encoder_hidden = encoder.initHidden(batch_size)
                EOS_flag_list = [None] * batch_size
                for i in range(batch_size):
                    EOS_flag_list[i] = False

                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(
                        input_variable[ei], encoder_hidden)

                start_token = np.ones((1, batch_size)) * SOS_token
                decoder_input = Variable(torch.LongTensor(start_token))
                decoder_input = decoder_input.cuda(0) if use_cuda else decoder_input

                decoder_hidden = encoder_hidden

                loss = 0

                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, None)

                    #loss = self.criterion(decoder_output[0], target_variable[di][0])
                    #losses.append(loss.data[0].item())

                    topv, topi = decoder_output.data.topk(1)
                    ni = topi.view(1,-1)
                    decoder_input = v2c(Variable(ni))
                    for i in range(batch_size):

                        if EOS_flag_list[i]:
                            decoder_input[0][i].data[0] = PAD_token
                            word_index = PAD_token
                        else: 
                            word_index = ni[0][i].item()


                        if(word_index == EOS_token) or (word_index == PAD_token):
                            EOS_flag_list[i] = True
                            continue
                        decoder_word = self.lang_txt.index2word[word_index]

                        if(batch_sentence[i] == ''):
                            batch_sentence[i] += (decoder_word)
                        else:
                            batch_sentence[i] += (' ' + decoder_word)
                if(see):
                    print(batch_sentence)
                    print(target_sentence)
                    print('#######################################')
                rewards = self.get_reward(batch_sentence, target_sentence, EOS_flag_list, types = 2)
                reward = np.mean(rewards)
                f_rs.append(reward)
                if(self.flag_god):
                    print(rewards)
            f_r = np.mean(f_rs)

            return f_r, 0

    
    def parameter_line(self, model1, model2, alpha = 0.1):
        model1_para = list(model1.parameters())
        model2_para = list(model2.parameters())
        for i, par in enumerate(model1.parameters()):
            par.data = alpha * model2_para[i].data + (1.0-alpha) * par.data
        return model1

  
    def random_sample(self,tensor):

        tensor1 = (torch.exp(tensor[0])).data.cpu().numpy()
        a,b = tensor1.shape
        samples = np.zeros((1, a))
        probs = np.zeros((1, a))
        for i in range(a):
            max_index = tensor1[i].argmax()
            tensor1[i, max_index] += (1 - tensor1[i].sum())

            index = np.random.choice(b, 1, p = tensor1[i])
            samples[:, i] = index
            probs[:, i] = tensor1[i,index]
        samples = v2c(Variable(torch.LongTensor(samples)))
        probs = v2c(Variable(torch.FloatTensor(probs)))
        return samples, probs

    def get_reward(self, batch_sentence, target_sentence, EOS_flag_list, gate = True, types = 1):

        if(self.task == 'counting'):
            if(self.flag_god):
                print('generate:',batch_sentence)
                print('target', target_sentence)
                print('********************************')
            reward = []
            for index, sentence in enumerate(batch_sentence):

                #if (EOS_flag_list[index] or gate):
                #bleu = int(sentence == target_sentence[index])
                #else:
                #    bleu = 0
                sen_list = sentence.split()
                target_list = target_sentence[index].split()

                bleu = 0.0
                if len(sen_list) == 3:
                    if int(sen_list[0]) + int(sen_list[2]) + 1 == len(target_list) and int(sen_list[0])>=0:
                        if sen_list[1] == target_list[int(sen_list[0])]:
                            bleu = 1.0
                reward.append(bleu)
            return (reward)
        elif(self.task == 'real'):
            reward = []
            for index, sentence in enumerate(batch_sentence):

                if sentence == '':
                    sentence = '.'
                if (EOS_flag_list[index] or gate):
                    if(types == 1):
                        bleu = BLEU([sentence], [target_sentence[index]], n = 2)
                    elif(types == 2):
                        bleu = BLEU([sentence], target_sentence[index], n = 2)

                else:
                    bleu = 0
                reward.append(bleu)
            return (reward)

    def fit_reward(self, reward, predict, lr = 0.0001):
       
        evaluater_optimizer = optim.SGD(self.evaluater.parameters(), lr=lr)

        evaluater_optimizer.zero_grad()

        batch_size = len(reward)
        sentence_length = len(predict)

        reward = v2c(Variable(torch.FloatTensor(reward)).view(-1, 1))
        reward_list = v2c(Variable(torch.ones(batch_size, sentence_length))) * reward

        predict_torch = v2c(Variable(torch.zeros(batch_size, sentence_length)))
        for index, pre in enumerate(predict):
            predict_torch[:,index] = pre.view(-1)

        fit_bleu_loss = self.evaluater_criterion(predict_torch, reward_list)
        fit_bleu_loss.backward(retain_graph = True)

        evaluater_optimizer.step()
  
    def fit_reward_adv(self, reward, predict, lr = 0.0001):

        evaluater_optimizer = optim.SGD(self.evaluater.parameters(), lr=lr)

        evaluater_optimizer.zero_grad()

        batch_size = predict.shape[0]
        sentence_length = predict.shape[1]

        #reward = v2c(Variable(torch.FloatTensor(reward)).view(-1, 1))
        reward_list = v2c(Variable(torch.ones(batch_size, sentence_length))) * reward


        fit_bleu_loss = self.evaluater_criterion(predict, reward_list)
        fit_bleu_loss.backward(retain_graph = True)

        evaluater_optimizer.step()


   
    def train(self,
              input_variable,
              target_variable,
              target_sentence, 
              reinforce_step,
              max_length=150,
              actor_fixed = False,
              train_method = 'XENT',
              use_ppo = False,
              ppo_b1 = 0.2, 
              ppo_b2 = 0.3, 
              ppo_a1 = 0.2,
              ppo_a2 = 0.05,
              ppo_a3 = 0.0,
              rate = 1.0,
              lr = 0.0001
              ):
        '''
        the training process of one batch

        input:
          input_varuable: the input of the encoder. It is a list, each element of
                          the list is one input of RNN unit. The length of list means the sentence length.
              type: pytorch LongTensor variable list
              shape: 
                  list length: input sentence length
                  element shape: (1, batch_size)
          target variable: the input of the decoder.It is a list, each element of
                           the list is one input of RNN unit. The length of list means the sentence length.
              type: pytorch LongTensor variable list
              shape: 
                  list length: input sentence length
                  element shape: (1, batch_size)
          target_sentence: the raw sentence of the target output in the training set(the content is same as target_variable)
                           In order to get the reward(count the BLEU score or accuracy) of the output.
              type:sentence list
              shape:
                  list length: batch_size
                  element: sentence
          reinforce_step: parameter. The length needed to execute reinforce algorithm
          max_length: parameter
          actor_fixed: parameter. If False, the encoder and decoder will not be updated
          train_method: parameter. Select different training type. We have: XENT, MIXED
          use_ppo: parameter. If True, use ppo method.
          ppo_b1, ppo_b2, ppo_a1, ppo_a2, ppo_a3, rate: parameters. PPO-dynamic parameter
          lr: parameter. Learning rate of encoder and deocder

        output:
          average loss
          average reward
        '''
        reward_ave = 0

        encoder = self.encoder
        decoder = self.decoder  
        encoder_prev = self.encoder_prev
        decoder_prev = self.decoder_prev
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

        criterion = self.criterion

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()


        batch_size = input_variable[0].size()[1]

        encoder_hidden = encoder.initHidden(batch_size)
        encoder_hidden_prev = encoder_prev.initHidden(batch_size)

        input_length = len(input_variable)
        target_length = len(target_variable)
        sentence_step = target_length - reinforce_step
        if sentence_step < -self.sentence_delta:
            sentence_step = -self.sentence_delta


        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
              input_variable[ei], encoder_hidden)
            _, encoder_hidden_prev = encoder_prev(
              input_variable[ei], encoder_hidden_prev)
            #encoder_outputs[ei] = encoder_output[0][0]

        start_token = np.ones((1, batch_size)) * SOS_token
        decoder_input = Variable(torch.LongTensor(start_token))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden
        decoder_hidden_prev = encoder_hidden_prev
        #use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False


        if train_method == 'MIXED':

            batch_sentence = [None] * batch_size
            for i in range(batch_size):
                batch_sentence[i]  = ''



            EOS_flag_list = [None] * batch_size
            for i in range(batch_size):
                EOS_flag_list[i] = False


            loss_matrix = v2c(Variable(torch.zeros(batch_size, target_length - sentence_step)))
            p_old = v2c(Variable(torch.zeros(batch_size, target_length - sentence_step)))
            reward_predict_matrix = v2c(Variable(torch.zeros(batch_size, target_length - sentence_step)))

            for di in range(sentence_step):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                  decoder_input, decoder_hidden, None)
                _, decoder_hidden_prev, _ = decoder_prev(
                  decoder_input, decoder_hidden_prev, None)
                #loss +=(criterion(decoder_output[0], target_variable[di][0]))

                #reward_predict = self.evaluater(decoder_hidden[0])
                #reward_predict_list.append(reward_predict)

                decoder_input = target_variable[di]  # Teacher forcing

                for i in range(batch_size):
                    word_index = decoder_input[0][i].item()
                    if(word_index == EOS_token) or (word_index == PAD_token) :
                        EOS_flag_list[i] = True
                        continue
                    decoder_word = self.lang_txt.index2word[word_index]
                    batch_sentence[i] += (' ' + decoder_word)



            for di in range(0, target_length - sentence_step):

                decoder_output_prev, decoder_hidden_prev, _ = decoder_prev(
                  decoder_input, decoder_hidden_prev, None)

                decoder_output, decoder_hidden, decoder_attention = decoder(
                  decoder_input, decoder_hidden, None)
                #topv, topi = decoder_output.data.topk(1)
                if use_ppo:
                    samples, probs = self.random_sample(decoder_output_prev)
                else:
                    samples, probs = self.random_sample(decoder_output)
                ni = (samples)
                decoder_input = ni.view(1, -1)

                END_flag = 0
                for i in range(batch_size):

                    if EOS_flag_list[i] == True:
                        decoder_input[0][i].data[0] = PAD_token
                        word_index = PAD_token
                    else: 
                        word_index = ni[0][i].item()
                  

                    if use_ppo:
                        #loss_list[i].append(torch.exp(decoder_output[0][i][word_index]))
                        loss_matrix[i, di] = (decoder_output[0][i][word_index])
                        p_old[i, di] = (decoder_output_prev[0][i][word_index].data)
                        #print(word_index)
                    else:
                        #word = v2c(Variable(torch.LongTensor([word_index])))
                        #  loss_list[i].append(criterion(decoder_output[0][i].view(1,-1), word))
                        loss_matrix[i, di] = decoder_output[0][i][word_index]
                    #else:
                    #loss_list[i].append(criterion(decoder_output[0][i].view(1,-1), target_variable[di][0][i].view(1)))

                    if(word_index == EOS_token) or (word_index == PAD_token):
                        END_flag +=1
                        EOS_flag_list[i] = True
                        continue
                    decoder_word = self.lang_txt.index2word[word_index]
                    batch_sentence[i] += (' ' + decoder_word)

                reward_input = v2c(Variable(decoder_hidden[0].data))
                reward_predict = self.evaluater(reward_input)
                #reward_predict_list.append(reward_predict)
                reward_predict_matrix[:, di] = reward_predict.view(-1)

                if END_flag == batch_size:
                    break  

            #set_trace()    
            reward = self.get_reward(batch_sentence, target_sentence, EOS_flag_list)
            reward_ave = np.array(reward).mean()

            reward = v2c(Variable(torch.FloatTensor(reward).view(-1,1)))
            if not use_ppo:
                #for index_batch, loss_trac in enumerate(loss_list):
                #    for index_len, loss_word in enumerate(loss_trac):
                #        loss += (reward[index_batch] - reward_predict_list[index_len ][index_batch]) * loss_word 
                      #set_trace()
                adv_matrix = reward - reward_predict_matrix
                loss = - (adv_matrix * loss_matrix).sum()

            else:
                #for index_batch, loss_trac in enumerate(loss_list):
                #    for index_len, loss_word in enumerate(loss_trac):

                #p_old = v2c(Variable(loss_matrix.data)) + 1e-4
                ratio = torch.exp(loss_matrix - p_old)
                p_old = torch.exp(p_old) + 1e-4
                adv_matrix = reward - reward_predict_matrix
                ratio_clip = v2c(Variable(torch.zeros(batch_size, target_length - sentence_step)))
                for index_batch in range(batch_size):
                    for index_sent in range(target_length - sentence_step):
                        #set_trace()
                        clip_b = 1 - (rate ** index_sent) * min(ppo_b1, ppo_b2 * (1/(p_old[index_batch, index_sent].data[0]) - 1 + 1e-3) ** 0.5)
                        clip_a = 1 +(rate ** index_sent) * min( ppo_a2 * (1/(p_old[index_batch, index_sent].data[0]) - 1 + 1e-3)**0.5 + ppo_a3 / (p_old[index_batch, index_sent].data[0]), ppo_a1)
                        ratio_clip[index_batch, index_sent] = torch.clamp(ratio[index_batch, index_sent], clip_b, clip_a)

                loss = -torch.min(ratio_clip * adv_matrix, ratio * adv_matrix).sum()

            self.fit_reward_adv(reward, reward_predict_matrix)
            #print('adv',adv_matrix)
            #print('reward', )

            if(self.flag_god):
                print('reward', reward, reward_predict_list[0])
                print('sentence\n', batch_sentence)
                print(target_sentence)
                print('#################################################')




        elif train_method == 'XENT':

            if True:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                      decoder_input, decoder_hidden, None)
                    loss += criterion(decoder_output[0], target_variable[di][0])
                    decoder_input = target_variable[di]  # Teacher forcing

        if not(actor_fixed):
            loss.backward()

            encoder_prev.load_state_dict(encoder.state_dict())
            decoder_prev.load_state_dict(decoder.state_dict())
            encoder_optimizer.step()
            decoder_optimizer.step()


        if train_method == 'XENT': 
            return loss.data[0] / (target_length), reward_ave
        elif train_method == 'MIXED':
            return loss.data[0] / (reinforce_step), reward_ave


    def trainIters(self,
                    batch_data, 
                    XENT_iters, 
                    R_iters,
                    actor_fixed = False,  
                    min_rein_step = 0, 
                    max_rein_step = 0, 
                    use_ppo = False,
                    ppo_b1 = 0.3, 
                    ppo_b2 = 0.3, 
                    ppo_a1 = 0.3,
                    ppo_a2 = 0.05,
                    ppo_a3 = 10.0, 
                    rate = 1.0,
                    lr = 0.001,
                    use_lr_decay = False,
                    decay_rate = 0.8,
                    print_every= 100, 
                    plot_every= 250,
                    dev_every = 300,
                    name = '_count_REINFORCE_PPO',
                    file_name = '~'):
      
        '''
        the whole training process.

        input: 
          batch_data: the whole training data.
              type: list
              length: the number of batches
              element: one batch of data
                  type: list
                  length: 3
                  element: input_variable, target_variable, target_sentence(same as the input in def train)
          XENT_iters: parameter. The iteration number of MLE.
          R_iters: parameter. The iteration number of each reinforce step length.
          min_rein_step: the minimun reinforce step length.
          max_rein_step: the maximum reinforce step length.
          use_lr_decay: parameter. If True , use learning rate decay mathod.
          decay_rate: parameter,The rate of learning_rate decay.
          dev_every: use the dev set to test the model every dev_every times.
          print_every: print the result every print_every times.
          plot_every: plot the result every print_every times
        '''

        self.lr = lr
        start = time.time()
        plot_losses = []
        plot_rs = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        print_r_total = 0
        plot_r_total = 0

        #name_encoder , name_decoder, name_evaluater, _, _ =  generate_save_text(self.encoder, self.decoder, self.evaluater, self.eva_encoder, self.eva_decoder)

        self.max_sentence_length = len(batch_data[0][0])

        training_count = 0

        n_iters = 0
        t_iters = XENT_iters + R_iters * (max_rein_step - min_rein_step + 1)
        for reinforce_step in range(min_rein_step, max_rein_step + 1):
            if reinforce_step == 0:
                for iter in range(1, XENT_iters + 1):
                    n_iters += 1
                    for batch in batch_data:

                        training_in, target, target_sentence = batch

                        loss, _ = self.train(
                              training_in,
                              target, 
                              target_sentence, 
                              reinforce_step,
                              train_method = 'XENT',
                              actor_fixed = actor_fixed,
                              lr = self.lr)

                        print_loss_total += loss
                        plot_loss_total += loss
                        training_count += 1

                        if training_count%print_every == 0:
                            print_loss_avg = print_loss_total / print_every
                            print_loss_total = 0
                            print('%s (%d %d%%) %.4f' % (timeSince(start, n_iters / t_iters),
                                                         n_iters, n_iters / t_iters * 100, print_loss_avg))
                            #plt.close()
                            #plt.plot(plot_losses)
                            #plt.show()
                            #torch.save(encoder.state_dict(), name_encoder)
                            #print('save encoder success')
                            #torch.save(decoder.state_dict(), name_decoder)
                            #print('save decoder success')
                            #torch.save(evaluater.state_dict(), name_evaluater)

                        if training_count % plot_every == 0:
                            plot_loss_avg = plot_loss_total / plot_every
                            plot_losses.append(plot_loss_avg)
                            self.god_loss.append(plot_loss_avg)
                            plot_loss_total = 0


                        if training_count%dev_every == 0:
                            #self.writer.add_scalar('data/scalar2', self.dev_score(self.dev_data), training_count/dev_every)
                            scoree, losses = self.dev_score(self.dev_data)
                            self.god_rs_dev.append(scoree)
                            self.god_loss_dev.append(losses)
                            #if(len(self.god_loss_dev)>10):
                            #  if((self.god_loss_dev[-1] + self.god_loss_dev[-2] + self.god_loss_dev[-3] + self.god_loss_dev[-4]) > (self.god_loss_dev[-5] + self.god_loss_dev[-6] + self.god_loss_dev[-7] + self.god_loss_dev[-8])):
                            #    self.lr_a *= decay_rate
                            print('reward', scoree)
                            print('loss', losses)
                            #torch.save(encoder.state_dict(), 'encoder_count_MLE' + str(training_count/dev_every))
                            #torch.save(decoder.state_dict(), 'decoder_count_MLE' + str(training_count/dev_every))

                            #plt.close()
                            #plt.plot(self.god_rs_dev)
                            #plt.show()

            else:
                print('start REINFORCE......step = ', reinforce_step)
                for iter in range(1, R_iters + 1):
                    n_iters += 1
                    for batch in batch_data:

                        training_in, target, target_sentence = batch

                        loss, reward = self.train(
                              training_in,
                              target, 
                              target_sentence, 
                              reinforce_step,
                              train_method = 'MIXED',
                              actor_fixed = actor_fixed, 
                              use_ppo = use_ppo,
                               ppo_b1 = ppo_b1, 
                             ppo_b2 = ppo_b2, 
                             ppo_a1 = ppo_a1,
                             ppo_a2 = ppo_a2,
                             ppo_a3 = ppo_a3, 
                              rate = rate, 
                              lr = self.lr)

                        print_loss_total += loss
                        plot_loss_total += loss
                        print_r_total += reward
                        plot_r_total += reward
                        training_count += 1

                        if training_count%print_every == 0:
                            print_loss_avg = print_loss_total / print_every
                            print_r_avg = print_r_total / print_every
                            print_loss_total = 0
                            print_r_total = 0
                            print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, n_iters / t_iters),
                                                         n_iters, n_iters / t_iters * 100, print_loss_avg, print_r_avg))
                            #torch.save(encoder.state_dict(), name_encoder)
                            #print('save encoder success')
                            #torch.save(decoder.state_dict(), name_decoder)
                            #print('save decoder success')

                        if training_count % plot_every == 0:
                            plot_loss_avg = plot_loss_total / plot_every
                            plot_losses.append(plot_loss_avg)
                            plot_r_avg = plot_r_total / plot_every
                            plot_rs.append(plot_r_avg)
                            self.god_rs.append(plot_r_avg)
                            plot_loss_total = 0
                            plot_r_total = 0

                        if training_count%dev_every == 0:
                            #self.writer.add_scalar('data/scalar2', self.dev_score(self.dev_data), training_count/dev_every)
                            scoree, losses = self.dev_score(self.dev_data)
                            scoree_test, _ = self.dev_score(self.test_data)
                            self.god_rs_dev.append(scoree)
                            self.god_rs_test.append(scoree_test)
                            self.god_loss_dev.append(losses)
                            if(len(self.god_rs_dev) > 2):
                                if(self.god_rs_dev[-1] < self.god_rs_dev[-2]):
                                    self.lr *= decay_rate
                            #if(len(self.god_loss_dev)>10):
                            #  if((self.god_loss_dev[-1] + self.god_loss_dev[-2] + self.god_loss_dev[-3] + self.god_loss_dev[-4]) > (self.god_loss_dev[-5] + self.god_loss_dev[-6] + self.god_loss_dev[-7] + self.god_loss_dev[-8])):
                            #    self.lr_a *= decay_rate
                            print('dev set reward', scoree)
                            print('dev set loss', losses)
                            #torch.save(encoder.state_dict(), file_name + '/encoder_' + name + str(training_count/dev_every))
                            #torch.save(decoder.state_dict(), file_name + '/decoder_' + name + str(training_count/dev_every))
                            #plt.close()
                            #plt.plot(self.god_rs_dev)
                            #plt.plot(self.god_rs_test)
                            #plt.show()

        return plot_losses, plot_rs



class ganSeq2seq(nn.Module):
  
    def __init__(self,lang_txt,
               dev_data, #dev set 
               test_data, #test set
               encoder,
               decoder,
               dis_encoder,
               dis_decoder,
               eva_encoder,
               eva_decoder,
               prev_encoder, #the encoder to record the previous state
               prev_decoder,
                 
               task = 'counting',
               max_length = MAX_LENGTH,
               god_g = [],
               god_d = [],
               god_e = [],
               god_rs_dev = [],
               god_loss_dev = [],
               god_loss = [],
               god_rs_test = [],
    ):
        super(ganSeq2seq,self).__init__()
        
        self.task = task
        self.lang_txt = lang_txt
        self.encoder = encoder
        self.decoder = decoder
        self.dis_encoder = dis_encoder
        self.dis_decoder = dis_decoder
        self.prev_decoder = prev_decoder
        self.prev_encoder = prev_encoder

        self.eva_encoder = eva_encoder
        self.eva_decoder = eva_decoder

        self.learning_rate = 0.001

        self.hidden_size = encoder.hidden_size

        self.criterion = nn.NLLLoss()
        self.criterion_q = nn.MSELoss()

        self.encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(decoder.parameters(), lr=self.learning_rate)
        self.dis_encoder_optimizer = optim.Adam(dis_encoder.parameters(), lr=self.learning_rate)
        self.dis_decoder_optimizer = optim.Adam(dis_decoder.parameters(), lr=self.learning_rate)

        self.eva_encoder_optimizer = optim.Adam(dis_encoder.parameters(), lr=self.learning_rate)
        self.eva_decoder_optimizer = optim.Adam(dis_decoder.parameters(), lr=self.learning_rate)

        self.print_every = 2
        self.plot_every = 2

        self.god_g = god_g
        self.god_d = god_d
        self.god_e = god_e

        self.dev_data = dev_data
        self.test_data = test_data
        self.god_rs_dev = god_rs_dev
        self.god_loss_dev = god_loss_dev
        self.god_loss = god_loss
        self.god_rs_test = god_rs_test

        self.flag_god = False

    def get_reward(self, batch_sentence, target_sentence, EOS_flag_list, gate = True, types = 1):

        if(self.task == 'counting'):
            if(self.flag_god):
                print('generate:',batch_sentence)
                print('target', target_sentence)
                print('********************************')
            reward = []
            for index, sentence in enumerate(batch_sentence):

                #if (EOS_flag_list[index] or gate):
                #bleu = int(sentence == target_sentence[index])
                #else:
                #    bleu = 0
                sen_list = sentence.split()
                target_list = target_sentence[index].split()

                bleu = 0.0
                if len(sen_list) == 3:
                    if int(sen_list[0]) + int(sen_list[2]) + 1 == len(target_list) and int(sen_list[0])>=0:
                        if sen_list[1] == target_list[int(sen_list[0])]:
                            bleu = 1.0
                reward.append(bleu)
            return (reward)
        elif(self.task == 'real'):
            reward = []
            for index, sentence in enumerate(batch_sentence):

                if sentence == '':
                    sentence = '.'
                if (EOS_flag_list[index] or gate):
                    bleu = BLEU([sentence], [[target_sentence[index]]])
                else:
                    bleu = 0
                reward.append(bleu)
            return (reward)
        
    def dev_score(self, dev_data, types = 1):
        '''
        get the score of the model.
        input: dev set / test set data
          type: data_type
        output: (average reward, average losses)
        '''
        
        if(types == 1):
            #dev_data = self.dev_data
            f_rs = []
            losses = []
            encoder = self.encoder
            decoder = self.decoder

            for index,batch in enumerate(dev_data):
                input_variable, target_variable, target_sentence = batch

                batch_size = input_variable[0].size()[1]
                input_length = len(input_variable)
                target_length = len(target_variable)
                batch_sentence = [None] * batch_size
                for i in range(batch_size):
                    batch_sentence[i]  = ''

                encoder_hidden = encoder.initHidden(batch_size)
                EOS_flag_list = [None] * batch_size
                for i in range(batch_size):
                    EOS_flag_list[i] = False

                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(
                        input_variable[ei], encoder_hidden)

                start_token = np.ones((1, batch_size)) * SOS_token
                decoder_input = Variable(torch.LongTensor(start_token))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                decoder_hidden = encoder_hidden

                loss = 0

                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, None)

                    loss = self.criterion(decoder_output[0], target_variable[di][0])
                    losses.append(loss.data[0])

                    topv, topi = decoder_output.data.topk(1)
                    ni = topi.view(1,-1)
                    decoder_input = v2c(Variable(ni))
                    for i in range(batch_size):

                        if EOS_flag_list[i]:
                            decoder_input[0][i].data[0] = PAD_token
                            word_index = PAD_token
                        else: 
                            word_index = ni[0][i].item()


                        if(word_index == EOS_token) or (word_index == PAD_token):
                            EOS_flag_list[i] = True
                            continue
                        decoder_word = self.lang_txt.index2word[word_index]

                        if(batch_sentence[i] == ''):
                            batch_sentence[i] += (decoder_word)
                        else:
                            batch_sentence[i] += (' ' + decoder_word)
                rewards = self.get_reward(batch_sentence, target_sentence, EOS_flag_list, types = types)
                reward = np.mean(rewards)
                f_rs.append(reward)
                if(self.flag_god):
                    print(rewards)
            f_r = np.mean(f_rs)

            return f_r, np.mean(losses)
        
        elif(types == 2):
            #dev_data = self.dev_data
            f_rs = []
            losses = []
            encoder = self.encoder
            decoder = self.decoder

            for index,batch in enumerate(dev_data):
                input_variable, target_variable, target_sentence = batch

                batch_size = input_variable[0].size()[1]
                input_length = len(input_variable)
                target_length = MAX_LENGTH
                batch_sentence = [None] * batch_size
                for i in range(batch_size):
                    batch_sentence[i]  = ''

                encoder_hidden = encoder.initHidden(batch_size)
                EOS_flag_list = [None] * batch_size
                for i in range(batch_size):
                    EOS_flag_list[i] = False

                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(
                        input_variable[ei], encoder_hidden)

                start_token = np.ones((1, batch_size)) * SOS_token
                decoder_input = Variable(torch.LongTensor(start_token))
                decoder_input = decoder_input.cuda(0) if use_cuda else decoder_input

                decoder_hidden = encoder_hidden

                loss = 0

                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, None)

                    #loss = self.criterion(decoder_output[0], target_variable[di][0])
                    #losses.append(loss.data[0].item())

                    topv, topi = decoder_output.data.topk(1)
                    ni = topi.view(1,-1)
                    decoder_input = v2c(Variable(ni))
                    for i in range(batch_size):

                        if EOS_flag_list[i]:
                            decoder_input[0][i].data[0] = PAD_token
                            word_index = PAD_token
                        else: 
                            word_index = ni[0][i].item()


                        if(word_index == EOS_token) or (word_index == PAD_token):
                            EOS_flag_list[i] = True
                            continue
                        decoder_word = self.lang_txt.index2word[word_index]

                        if(batch_sentence[i] == ''):
                            batch_sentence[i] += (decoder_word)
                        else:
                            batch_sentence[i] += (' ' + decoder_word)
                if(see):
                    print(batch_sentence)
                    print(target_sentence)
                    print('#######################################')
                rewards = self.get_reward(batch_sentence, target_sentence, EOS_flag_list, types = 2)
                reward = np.mean(rewards)
                f_rs.append(reward)
                if(self.flag_god):
                    print(rewards)
            f_r = np.mean(f_rs)

            return f_r, 0

    def parameter_line(self, model1, model2, alpha = 0.1):
        model1_para = list(model1.parameters())
        model2_para = list(model2.parameters())
        for i, par in enumerate(model1.parameters()):
            par.data = alpha * model2_para[i].data + (1.0-alpha) * par.data
        return model1

  
    def random_sample(self,tensor, width = 1):
        
        '''
        get a batch of random samples of some width from a distribution
        
        inputs:
          tensor:log porbability of a batch of words index
              shape: 1 x batch_size x length_of_words
              type:torch variable
          width:the number of nodes expanded
        outputs:
          samples_t:samples of words index from prob distribution
              shape: width x 1 x batch_size
          probs_t:probabilities of samples of words index from prob distribution
              shape: width x 1 x batch_size
        '''
        
        tensor1 = (torch.exp(tensor[0])).data.cpu().numpy()
        a,b = tensor1.shape
    
        samples_t = np.zeros((width, 1, a))
        
        probs_t = np.zeros((width, 1, a))
        for i in range(a):
            max_index = tensor1[i].argmax()
            tensor1[i, max_index] += (1 - tensor1[i].sum())
            index = np.random.choice(b, width, p = tensor1[i], replace = False)
            for iter in range(width):
                samples_t[iter, :, i] = index[iter]
                probs_t[iter, :, i] = tensor1[i,index[iter]]
        samples_t = v2c(Variable(torch.LongTensor(samples_t)))
        probs_t = v2c(Variable(torch.FloatTensor(probs_t)))
        return samples_t, probs_t

    def mc_search(self,
              samples_t, 
              mc_length,
              decoder_hidden_mc,
              dis_decoder_hidden_mc,
              search_n = 3):
  
        '''
        take the monte carlo search alog

        input:
          samples_t: sampled word index data
              size: width x 1 x batch_size
              type: torch variable
          mc_length:monte carlo step length(should bigger than 0)
              type: int
          decoder_hidden_mc dis_decoder_hidden_mc: hidden state of current
              type: torch variable
          search_n: monte carlo search width
              type: int

        output:
          q_score: the estimated q_value of each state
              size: width x batchsize
              type: torch data
        '''
        
        width = samples_t.shape[0]
        batchsize = samples_t.shape[2]

        q_score = v2c(Variable(torch.zeros(width, batchsize)))

        for iter_width in range(width):
            decoder_input_mc = samples_t[iter_width]
            for iter_n in range(search_n):
                for di in range(mc_length):
                    decoder_output_mc, decoder_hidden_mc, _ = self.prev_decoder(
                          decoder_input_mc, decoder_hidden_mc, None) 
                    dis_decoder_output_mc, dis_decoder_hidden_mc, _ = self.dis_decoder(
                          decoder_input_mc, dis_decoder_hidden_mc, None)

                    samples, probs = self.random_sample(decoder_output_mc, width = 1)
                    ni = samples.view(1,-1)
                    decoder_input_mc = ni
                q_score[iter_width] = q_score[iter_width] + dis_decoder_output_mc.view(-1)

        q_score /= search_n
        return q_score.data
    
    def get_prob_from_index(self, samples_t, decoder_output, return_data = False):
    
        '''
        input: 
            samples_t: sampled index of a batch of words
              shape: width x 1 x batchsize
              type: torch float
            decoder_output: the output of decoder layer
              shape: 1 x batchsize x words_length
              type:torch variable

        output:
            log_p: sampled log_propability of a batch of words
              shape: width x batchsize
              type: torch variable
        '''

        width = samples_t.shape[0]
        batchsize = samples_t.shape[2]

        log_p = v2c(Variable(torch.zeros(width, batchsize)))

        for iter_w in range(width):
            for iter_b in range(batchsize):
                index = samples_t[iter_w, 0, iter_b].data[0]
                log_p[iter_w, iter_b] = decoder_output[0, iter_b, index]

        if return_data:
            return v2c(Variable(log_p.data))
        else:
            return log_p
   
    def train(self,
            input_variable, 
            target_variable,
            train_method = 'G',
            search_n = 3,
            width = 3,
            g_lr = 0.001, 
            d_lr = 0.001, 
            e_lr = 0.001,
            use_ppo = False,
             ppo_b1 = 0.3, 
              ppo_b2 = 0.3, 
              ppo_a1 = 0.3,
              ppo_a2 = 0.05,
              ppo_a3 = 10
            ):
            
        encoder = self.encoder
        decoder = self.decoder    
        dis_encoder = self.dis_encoder
        dis_decoder = self.dis_decoder  
        eva_encoder = self.eva_encoder
        eva_decoder = self.eva_decoder
        prev_decoder = self.prev_decoder    
        prev_encoder = self.prev_encoder

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=g_lr)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=g_lr)
        dis_encoder_optimizer = optim.Adam(dis_encoder.parameters(), lr=d_lr)
        dis_decoder_optimizer = optim.Adam(dis_decoder.parameters(), lr=d_lr)

        eva_encoder_optimizer = optim.Adam(eva_encoder.parameters(), lr=e_lr)
        eva_decoder_optimizer = optim.Adam(eva_decoder.parameters(), lr=e_lr)

        criterion = self.criterion

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        dis_encoder_optimizer.zero_grad()
        dis_decoder_optimizer.zero_grad()
        eva_encoder_optimizer.zero_grad()
        eva_decoder_optimizer.zero_grad()

        input_length = len(input_variable)
        target_length = len(target_variable)
        batch_size = input_variable[0].size()[1]

        encoder_hidden = encoder.initHidden(batch_size)
        prev_encoder_hidden = encoder.initHidden(batch_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
              input_variable[ei], encoder_hidden)

        decoder_hidden = encoder_hidden


        loss_g = 0
        loss_d = 0
        loss_q = 0

        if train_method == 'G':

            start_token = np.ones((1, batch_size)) * SOS_token
            decoder_input = Variable(torch.LongTensor(start_token))
            decoder_input = decoder_input.cuda(0) if use_cuda else decoder_input

            for di in range(target_length):
                decoder_output, decoder_hidden, _ = decoder(
                  decoder_input, decoder_hidden, None)
                loss_g += criterion(decoder_output[0], target_variable[di][0])
                decoder_input = target_variable[di] 

            loss_g.backward()

            prev_decoder.load_state_dict(decoder.state_dict())
            prev_encoder.load_state_dict(encoder.state_dict())
            encoder_optimizer.step()
            decoder_optimizer.step()

            return loss_g.data[0]/(target_length * batch_size)

        elif train_method == 'GD':
            dis_encoder_hidden = dis_encoder.initHidden(batch_size)
            eva_encoder_hidden = eva_encoder.initHidden(batch_size)

            for ei in range(input_length):
                _, dis_encoder_hidden = dis_encoder(
                input_variable[ei], dis_encoder_hidden)
                _, eva_encoder_hidden = eva_encoder(
                input_variable[ei], eva_encoder_hidden)

                _, prev_encoder_hidden = prev_encoder(
                input_variable[ei], prev_encoder_hidden)



            dis_decoder_hidden = dis_encoder_hidden
            eva_decoder_hidden = eva_encoder_hidden
            prev_decoder_hidden = prev_encoder_hidden

            start_token = np.ones((1, batch_size)) * SOS_token
            decoder_input = Variable(torch.LongTensor(start_token))
            decoder_input = decoder_input.cuda(0) if use_cuda else decoder_input
            dis_decoder_input = Variable(torch.LongTensor(start_token))
            dis_decoder_input = dis_decoder_input.cuda(0) if use_cuda else dis_decoder_input

            for di in range(target_length):
                prev_decoder_output, prev_decoder_hidden, _ = prev_decoder(
                        decoder_input, prev_decoder_hidden, None) 

                decoder_output, decoder_hidden, _ = decoder(
                        decoder_input, decoder_hidden, None) 


                dis_decoder_output, dis_decoder_hidden, _ = dis_decoder(
                        dis_decoder_input, dis_decoder_hidden, None)

                eva_decoder_output, eva_decoder_hidden, _ = eva_decoder(
                        dis_decoder_input, eva_decoder_hidden, None)


                samples_t, probs_t = self.random_sample(prev_decoder_output, width = width)
                #print('@@@@@@@', samples_t)
                dis_decoder_input = samples_t[0]
                decoder_input = samples_t[0]

                q_value = self.mc_search(samples_t, 
                                  mc_length = target_length - di,
                                  decoder_hidden_mc = prev_decoder_hidden,
                                  dis_decoder_hidden_mc = dis_decoder_hidden,
                                  search_n = search_n)
                q_value = v2c(Variable(q_value))

                q_predict = v2c(Variable(torch.zeros(width, batch_size)))
                for iter_width in range(width):
                    eva_decoder_output, _, _ = eva_decoder(
                        samples_t[iter_width], eva_decoder_hidden, None)
                  
                    q_predict[iter_width] = eva_decoder_output[0].view(-1)

                loss_q_predict = ((q_value - q_predict) ** 2).sum()/(width * batch_size)
                loss_q += loss_q_predict
                q_predict = v2c(Variable(q_predict.data))

                log_p = self.get_prob_from_index(samples_t, decoder_output)

                if use_ppo:
                    p_old = self.get_prob_from_index(samples_t, prev_decoder_output, return_data = True)
                    ratio = torch.exp(log_p - p_old)
                    p_old = torch.exp(p_old) + 1e-4

                    ratio_clip = v2c(Variable(torch.zeros(width, batch_size)))
                    for index_width in range(width):
                        for index_batch in range(batch_size):
                            #set_trace()
                            clip_b = 1 - min(ppo_b1, ppo_b2 * (1/(p_old[index_width, index_batch].data[0]) - 1 + 1e-3) ** 0.5)
                            clip_a = min(1 + ppo_a2 * (1/(p_old[index_width, index_batch].data[0]) - 1 + 1e-3)**0.5 + ppo_a3 / (p_old[index_width, index_batch].data[0]), ppo_a1)
                            ratio_clip[index_width, index_batch] = torch.clamp(ratio[index_width, index_batch], clip_b, clip_a)
                    loss_word = -torch.min(ratio_clip * (q_value - q_predict), ratio * (q_value - q_predict)).sum()
                    loss_g += loss_word
                else:
                    loss_word = -(log_p * (q_value - q_predict)).sum()
                    loss_g += loss_word

            loss_d = -torch.log(1 - dis_decoder_output[0] + 1e-4).sum()


            ##############################################################
            dis_encoder_hidden = dis_decoder.initHidden(batch_size)

            for ei in range(input_length):
                _, dis_encoder_hidden = dis_encoder(
                input_variable[ei], dis_encoder_hidden)

            dis_decoder_hidden = dis_encoder_hidden

            dis_decoder_input = Variable(torch.LongTensor(start_token))
            dis_decoder_input = dis_decoder_input.cuda(0) if use_cuda else dis_decoder_input

            for di in range(target_length):

                dis_decoder_output, dis_decoder_hidden, _ = dis_decoder(
                        dis_decoder_input, dis_decoder_hidden, None)
                dis_decoder_input = target_variable[di]

            loss_d += (-torch.log(dis_decoder_output[0] + 1e-4).sum())
            ############################################################


            loss_g.backward()
            loss_d.backward()
            loss_q.backward()

            prev_decoder.load_state_dict(decoder.state_dict())
            prev_encoder.load_state_dict(encoder.state_dict())

            encoder_optimizer.step()
            decoder_optimizer.step()
            dis_encoder_optimizer.step()
            dis_decoder_optimizer.step()
            eva_encoder_optimizer.step()
            eva_decoder_optimizer.step()

            self.god_e.append(loss_q)
            return loss_g.data[0]/(target_length * batch_size * width), loss_d.data[0]/(batch_size * 2)

        elif train_method == 'D':

            dis_encoder_hidden = dis_decoder.initHidden(batch_size)

            for ei in range(input_length):
                _, dis_encoder_hidden = dis_encoder(
                input_variable[ei], dis_encoder_hidden)

            dis_decoder_hidden = dis_encoder_hidden

            start_token = np.ones((1, batch_size)) * SOS_token
            decoder_input = Variable(torch.LongTensor(start_token))
            decoder_input = decoder_input.cuda(0) if use_cuda else decoder_input
            dis_decoder_input = Variable(torch.LongTensor(start_token))
            dis_decoder_input = dis_decoder_input.cuda(0) if use_cuda else dis_decoder_input

            for di in range(target_length):
                decoder_output, decoder_hidden, _ = decoder(
                        decoder_input, decoder_hidden, None) 


                dis_decoder_output, dis_decoder_hidden, _ = dis_decoder(
                        dis_decoder_input, dis_decoder_hidden, None)



                samples_t, probs_t = self.random_sample(decoder_output, width = width)

                dis_decoder_input = samples_t[0]

            loss_d = -torch.log(1 + 1e-4 - dis_decoder_output[0]).sum()


            ##############################################################
            dis_encoder_hidden = dis_decoder.initHidden(batch_size)

            for ei in range(input_length):
                _, dis_encoder_hidden = dis_encoder(
                input_variable[ei], dis_encoder_hidden)

            dis_decoder_hidden = dis_encoder_hidden

            dis_decoder_input = Variable(torch.LongTensor(start_token))
            dis_decoder_input = dis_decoder_input.cuda(0) if use_cuda else dis_decoder_input

            for di in range(target_length):

                dis_decoder_output, dis_decoder_hidden, _ = dis_decoder(
                        dis_decoder_input, dis_decoder_hidden, None)
                dis_decoder_input = target_variable[di]

            loss_d += (-torch.log(dis_decoder_output[0]).sum())
            ############################################################


            loss_d.backward()
            dis_encoder_optimizer.step()
            dis_decoder_optimizer.step()

            return loss_d.data[0]/(batch_size * 2)



    def trainIters(self,
                 batch_data, 
                 G_iters = 0,
                 D_iters = 0,
                 GD_iters = 0, 
                 use_ppo = False,
                 g_lr = 0.001,
                 d_lr = 0.001,
                 e_lr = 1e-4,
                  ppo_b1 = 0.3, 
                  ppo_b2 = 0.3, 
                  ppo_a1 = 0.3,
                  ppo_a2 = 0.05,
                  ppo_a3 = 10,
                 decay_rate = 0.7, 
                 search_n = 3,
                 width = 3,
                 print_every = 50,
                 plot_every = 25,
                 dev_every = 100
                ):
        start = time.time()
        plot_losses_g = []
        plot_losses_d = []
        print_loss_total_g = 0  # Reset every print_every
        print_loss_total_d = 0
        plot_loss_total_g = 0  # Reset every plot_every
        plot_loss_total_d = 0
        #print_every = self.print_every
        #plot_every = self.plot_every

        #name_encoder , name_decoder, name_dis_encoder, name_dis_decoder =  generate_save_text(self.encoder, self.decoder, self.dis_encoder, self.dis_decoder)   

        n_iters = 0
        t_iters = D_iters + G_iters + GD_iters

        for iter in range(1, G_iters + 1):
            n_iters += 1
            training_count = 0

            for it, batch in enumerate(batch_data):

                if(it % 200 == 0):
                    print('finished', (it + 0.0)/len(batch_data))
                    training_in, target, target_sentence = batch

                loss = self.train(
                  training_in,
                  target, 
                   g_lr = g_lr,
                  d_lr = d_lr,
                  e_lr = e_lr,
                  train_method = "G")

                print_loss_total_g += loss
                plot_loss_total_g += loss
                training_count += 1


                if training_count%print_every == 0:
                    print_loss_avg = print_loss_total_g / print_every
                    print_loss_total_g = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, n_iters / t_iters),
                                               n_iters, n_iters / t_iters * 100, print_loss_avg))


                if training_count % plot_every == 0:
                    plot_loss_avg = plot_loss_total_g / plot_every
                    plot_losses_g.append(plot_loss_avg)
                    plot_loss_total_g = 0

                if training_count%dev_every == 0:
                    #self.writer.add_scalar('data/scalar2', self.dev_score(self.dev_data), training_count/dev_every)
                    scoree, losses = self.dev_score(self.dev_data)
                    self.god_rs_dev.append(scoree)
                    self.god_loss_dev.append(losses)
                    print('reward', scoree)
                    print('loss', losses)
                    #torch.save(encoder.state_dict(), 'encoder_count_MLE' + str(training_count/dev_every))
                    #torch.save(decoder.state_dict(), 'decoder_count_MLE' + str(training_count/dev_every))
                    #plt.close()
                    #plt.plot(self.god_rs_dev)
                    #plt.plot(self.god_loss_dev)
                    #plt.show()


        print('start training D')

        for iter in range(1, D_iters + 1):
            n_iters += 1
            training_count = 0

            for it, batch in enumerate(batch_data):

                if(it % 200 == 0):
                    print('finished', (it + 0.0)/len(batch_data))
                training_in, target, target_sentence = batch

                loss = self.train(
                  training_in,
                  target, 
                   g_lr = g_lr,
                  d_lr = d_lr,
                  e_lr = e_lr,
                  train_method = "D")

                #print('aaa',loss, training_count)
                print_loss_total_d += loss
                plot_loss_total_d += loss
                training_count += 1


                if training_count%print_every == 0:
                    print_loss_avg = print_loss_total_d / print_every
                    print_loss_total_d = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, n_iters / t_iters),
                                               n_iters, n_iters / t_iters * 100, print_loss_avg))

                if training_count % plot_every == 0:
                    plot_loss_avg = plot_loss_total_d / plot_every
                    plot_losses_d.append(plot_loss_avg)
                    plot_loss_total_d = 0

        print('start training GD')

        for iter in range(1, GD_iters + 1):
            n_iters += 1
            training_count = 0

            for it, batch in enumerate(batch_data):

                if(it % 200 == 0):
                    print('finished', (it + 0.0)/len(batch_data))
                training_in, target, target_sentence = batch

                loss_g, loss_d = self.train(
                  training_in,
                  target, 
                  train_method = "GD",
                  use_ppo = use_ppo,
                  g_lr = g_lr,
                  d_lr = d_lr,
                  e_lr = e_lr,
                   ppo_b1 = ppo_b1, 
                   ppo_b2 = ppo_b2, 
                   ppo_a1 = ppo_a1,
                   ppo_a2 = ppo_a2,
                  ppo_a3 = ppo_a3,
                  search_n = search_n,
                  width = width)

                #print('aaa',loss, training_count)
                print_loss_total_g += loss_g
                plot_loss_total_g += loss_g
                print_loss_total_d += loss_d
                plot_loss_total_d += loss_d
                training_count += 1


                if training_count%print_every == 0:
                    print_loss_avg_d = print_loss_total_d / print_every
                    print_loss_total_d = 0
                    print_loss_avg_g = print_loss_total_g / print_every
                    print_loss_total_g = 0
                    print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, n_iters / t_iters),
                                               n_iters, n_iters / t_iters * 100, print_loss_avg_g, print_loss_avg_d))

                if training_count % plot_every == 0:
                    plot_loss_avg_g = plot_loss_total_g / plot_every
                    plot_losses_g.append(plot_loss_avg_g)
                    plot_loss_avg_d = plot_loss_total_d / plot_every
                    plot_losses_d.append(plot_loss_avg_d)
                    self.god_g.append(plot_loss_avg_g)
                    self.god_d.append(plot_loss_avg_d)

                    plot_loss_total_d = 0
                    plot_loss_total_g = 0

                if training_count%dev_every == 0:
                    #self.writer.add_scalar('data/scalar2', self.dev_score(self.dev_data), training_count/dev_every)
                    scoree, losses = self.dev_score(self.dev_data)
                    scoree_test, _ = self.dev_score(self.test_data)
                    self.god_rs_dev.append(scoree)
                    self.god_loss_dev.append(losses)
                    self.god_rs_test.append(scoree_test)
                    print('reward', scoree)
                    print('loss', losses)
                    if(len(self.god_loss_dev)>2):
                        if(self.god_loss_dev[-1]<self.god_loss_dev[-2]):
                            g_lr *= decay_rate
                            d_lr *= decay_rate
                    #torch.save(encoder.state_dict(), 'encoder_count_MLE' + str(training_count/dev_every))
                    #torch.save(decoder.state_dict(), 'decoder_count_MLE' + str(training_count/dev_every))
                    #plt.close()
                    #plt.plot(self.god_rs_dev)
                    #plt.plot(self.god_rs_test)
                    #plt.show()  
        return plot_losses_g, plot_losses_d
