import random
import numpy as np
import torch
import argparse
import time
import sys
import torch.nn.functional as F
import torch.nn as nn
import time
from training import *
from models import *
from process_data import *

def parse_argument_file(filepath):
    #default hyperparameter settings
    hyperparameters = {
    'teacher_force':True,
    'attention':True,
    'attention_type':'dot',
    'recurrent_type':'GRU',
    'embedding_size':16,
    'hidden_size':64,
    'num_epochs':50,
    'print_freq':5,
    'batch_size':100,
    'learning_rate':0.001,
    'dropout_prob':0.1
    }

    f = open(filepath, 'r')
    for line in f.readlines():
        line = line.strip()
        line = line.split('\t')
        if line[0] in hyperparameters.keys():
            data_type = type(hyperparameters[line[0]])
            hyperparameters[line[0]] = data_type(line[1])
        elif line[0] == 'input_file':
            data_path = line[1]
        else:
            print('Warning: the following line in the hyperparameter file was ignored:\n\t' + '\t'.join(line))

    return(data_path, hyperparameters)

def initialize_networks(hyperparameters, num_phones, in_length, out_length, device):
    encoder_params = {}
    encoder_params['inventory_size'] = num_phones
    encoder_params['embedding_size'] = hyperparameters['embedding_size']
    encoder_params['hidden_size'] = hyperparameters['hidden_size']
    encoder_params['batch_size'] = hyperparameters['batch_size']
    encoder_params['dropout_p'] = hyperparameters['dropout_prob']
    encoder_params['recurrence'] = hyperparameters['recurrent_type']

    decoder_params = {}
    decoder_params['inventory_size'] = num_phones
    decoder_params['embedding_size'] = hyperparameters['embedding_size']
    decoder_params['hidden_size'] = hyperparameters['hidden_size']
    decoder_params['dropout_p'] = hyperparameters['dropout_prob'] #optional dropout layer turned off by setting p to 0.0
    decoder_params['max_in_length'] = in_length
    decoder_params['max_out_length'] = out_length
    decoder_params['with_attn'] = hyperparameters['attention']
    decoder_params['attn_type'] = hyperparameters['attention_type']
    decoder_params['recurrence'] = hyperparameters['recurrent_type']

    encoder = EncoderRNN(encoder_params).to(device)
    decoder = DecoderRNN(decoder_params).to(device)

    return encoder, decoder


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_file, hyperparams = parse_argument_file(sys.argv[1])
    data, output_length, input_length = load_data(data_file)
    phone2ix, ix2phone, num_phones, trainX, devX, trainY, devY = process_data(data, output_length, input_length, device)

    encode_net, decode_net = initialize_networks(hyperparams, num_phones, input_length, output_length, device)

    batch_train(encode_net, decode_net, device, trainX, trainY, devX, devY, ix2phone,
        hyperparams['num_epochs'], bsz=hyperparams['batch_size'], 
        learning_rate=hyperparams['learning_rate'], teacher_forcing=hyperparams['teacher_force'],
        print_every=hyperparams['print_freq'])
    
    # print('\nFinal:')
    # print('Resubstitution accuracy: ' + str(batch_predict(X, Y, encode_net, decode_net, 2, final=True)))
    # print('Dev accuracy: ' + str(batch_predict(devX, devY, encode_net, decode_net, 2, final=True)))

    # end = time.clock()
    # print('Time: ' + str(end - start))