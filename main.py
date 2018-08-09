import model
import prepare
import train
from io import open
import os
import argparse
from random import shuffle
import torch
use_cuda = torch.cuda.is_available()


def main(config):

    if(config.dataset == 'real'):
        #initialize the dictionary
        lang_real = prepare.Lang_real('txt')
        lines = open('data/opensubtitles/vocab4000').read().strip().split('\n')
        for sen in lines:
            lang_real.addSentence(sen)
        lang_txt = lang_real

        train_data = prepare.get_dataset('data/opensubtitles/train.txt', batch_size = 16, lang_txt = lang_real, task = 'real')
        shuffle(train_data)
        dev_data = prepare.get_dataset('data/opensubtitles/dev.txt', batch_size = 16, lang_txt = lang_real, task = 'real')
        test_data = prepare.get_dataset('data/opensubtitles/test.txt', batch_size = 16, lang_txt = lang_real, task = 'real')
        
    elif(config.dataset == 'counting'):
        lang_counting = prepare.Lang_counting('txt')
        lang_txt = lang_counting

        train_data = prepare.get_dataset('data/counting/train_counting.txt', batch_size = 16, lang_txt = lang_counting, task = 'counting')
        shuffle(train_data)
        dev_data = prepare.get_dataset('data/counting/dev_counting.txt', batch_size = 16, lang_txt = lang_counting, task = 'counting')
        test_data = prepare.get_dataset_test_counting('data/counting/test_counting.txt', batch_size = 16)

    feature = config.feature
    encoder = model.EncoderRNN(feature, feature, lang_txt.n_words)                     
    decoder = model.DecoderRNN(feature, feature,  lang_txt.n_words)
    evaluater = model.EvaluateR(feature)
    decoder_prev =  model.DecoderRNN(feature, feature,  lang_txt.n_words)
    encoder_prev =  model.EncoderRNN(feature, feature,  lang_txt.n_words)
    dis_encoder = model.disEncoderRNN(feature, feature, lang_txt.n_words)
    dis_decoder = model.disDecoderRNN(feature, feature, lang_txt.n_words)
    eva_encoder = model.disEncoderRNN(feature, feature, lang_txt.n_words)
    eva_decoder = model.disDecoderRNN(feature, feature, lang_txt.n_words)
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        evaluater= evaluater.cuda()
        decoder_prev = decoder_prev.cuda()
        encoder_prev = encoder_prev.cuda()
        dis_encoder = dis_encoder.cuda(0)
        dis_decoder = dis_decoder.cuda(0)
        eva_encoder = eva_encoder.cuda(0)
        eva_decoder = eva_decoder.cuda(0)


    print_every = config.print_every
    dev_every = config.dev_every
    use_ppo = config.use_ppo
    ppo_a1 = config.ppo_a1
    ppo_a2 = config.ppo_a2
    ppo_b1 = config.ppo_b1
    ppo_b2 = config.ppo_b2

    if(config.type == 'reinforce'): 

        lr = config.lr
        test1 = train.seq2seq(lang_txt, dev_data,test_data,  encoder, decoder, evaluater, 
                   encoder_prev,decoder_prev,
                    task = config.dataset,
                   god_rs_dev = [],
                    god_loss_dev = [],
                    god_loss = [],
                    god_rs_test = [])

        losses, rewards = test1.trainIters(train_data,1,1,
                                           use_ppo = use_ppo,actor_fixed = False, 
                                           min_rein_step = 0, max_rein_step = 5,
                                          ppo_b1 = ppo_b1, 
                                          ppo_b2 = ppo_b2, 
                                          ppo_a1 = ppo_a1,
                                          ppo_a2 = ppo_a2,
                                           ppo_a3 = 1e10,
                                           rate = 1,
                                          lr = lr,
                                          dev_every = dev_every,
                                          print_every = print_every,
                                          plot_every = 5000000000,
                                          name = '_z',
                                          file_name = 'MIXER')
    elif(config.type == 'gan'):

        test_gan = train.ganSeq2seq(lang_txt, dev_data,test_data,
                encoder,decoder,
                dis_encoder,dis_decoder,
                eva_encoder, eva_decoder,
                encoder_prev,
                decoder_prev,
                god_rs_dev = [],
               god_loss_dev = [],
               god_loss = [],
               god_rs_test = [],
                task = config.dataset)

        loss_g, loss_d = test_gan.trainIters(train_data, 0,0,1,
                                  use_ppo= config.use_ppo,g_lr = config.g_lr, d_lr = config.d_lr,
                                  search_n = 1, width = 1,
                                 ppo_b1 = ppo_b1, 
                                  ppo_b2 = ppo_b2, 
                                  ppo_a1 = ppo_a1,
                                  ppo_a2 = ppo_a2, 
                                  ppo_a3 = 10000000000,
                                 print_every = print_every,
                                 plot_every = 50000000000,
                                 dev_every = dev_every)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'real', choices = ['real', 'counting'])
    parser.add_argument('--type', type = str, default = 'gan', choices = ['gan', 'reinforce'])

    parser.add_argument('--feature', type = int, default = 128, help = 'the size of GRU')
    parser.add_argument('--print_every', type = int, default = 5, help = 'print the result every # iteration')
    parser.add_argument('--dev_every', type = int, default = 5, help = 'test on the dev set every # iteration')
    #parser.add_argument('--feature', type = int, default = 128, help = 'the size of GRU')

    parser.add_argument('--XENT_iter', type = int, default = 5, help = 'iteration of MLE training')
    parser.add_argument('--REINFORCE_iter', type = int, default = 5, help = 'iteration of REINFORCE training')
    #parser.add_argument('--min_rein_step', type = int, default = 0, help = 'minimum REINFORCE step')
    parser.add_argument('--max_rein_step', type = int, default = 5, help = 'maximum REINFORCE step')

    parser.add_argument('--ppo_a1', type = float, default = 0.3, help = 'ppo_dynamic parameter')
    parser.add_argument('--ppo_a2', type = float, default = 0.3, help = 'ppo_dynamic parameter')
    parser.add_argument('--ppo_b1', type = float, default = 0.3, help = 'ppo_dynamic parameter')
    parser.add_argument('--ppo_b2', type = float, default = 0.3, help = 'ppo_dynamic parameter')

    parser.add_argument('--lr', type = float, default = 1e-4, help = 'learning_rate of encoder and decoder in reinforce')
    parser.add_argument('--g_lr', type = float, default = 1e-5, help = 'learning_rate of generator and decoder in gan')
    parser.add_argument('--d_lr', type = float, default = 1e-5, help = 'learning_rate of discriminator and decoder in gan')

    parser.add_argument('--use_ppo', type = bool, default = True, help = 'use ppo method or not')

    config = parser.parse_args()
    print(config)
    main(config)