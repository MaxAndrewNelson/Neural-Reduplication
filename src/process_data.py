import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pair2tensors(pair, phone2ix):
    in_word = '<' + pair[0] + '>'
    out_word = '<' + pair[1] + '>'
    in_index = [phone2ix[p] for p in in_word]
    out_index = [phone2ix[p] for p in out_word]

    return (torch.LongTensor(in_index), torch.LongTensor(out_index))

def load_data(file): 
    file = open(file, 'r')  
    data = []
    max_len=0
    max_in=0
    for line in file.readlines():
        line=line.rstrip()
        line = line.split('\t')
        data.append((line[0], line[2]))
        if len(line[2]) > max_len:
            max_len = len(line[2])
            max_in = len(line[0])
    file.close()
    
    return data, max_len+2, max_in+2 #adding 2 to the max values for the start and end symbols

def process_data(data, max_out_len, max_in_len, device):
    phones = ['<','>','#'] #end and padding tokens
    for pair in data:
        for word in pair:
            for phone in word:
                if phone not in phones:
                    phones.append(phone)
    #build dictionaries to index phonemes
    phone2ix = {phone:i for i,phone in enumerate(phones)}
    ix2phone = dict((v,k) for k,v in phone2ix.items())
    num_phones = len(phone2ix.keys()) 

    #convert data into tensors   
    as_tensors = [pair2tensors(tup, phone2ix) for tup in data]

    #pad data up to the maximum length
    X = [F.pad(pair[0],(0,max_in_len-len(pair[0])),value=phone2ix['#']) for pair in as_tensors]
    Y = [F.pad(pair[1],(0,max_out_len-len(pair[1])),value=phone2ix['#']) for pair in as_tensors]

    X = torch.stack(X).to(device)
    Y = torch.stack(Y).to(device)

    train_dev_prop = 0.7 #proportion of data in train, remaining goes in dev
    train_num = int(train_dev_prop*X.size()[0])

    trainX, devX = torch.split(X, train_num, dim=0)
    trainY, devY = torch.split(Y, train_num, dim=0)

    return(phone2ix, ix2phone, num_phones, trainX, devX, trainY, devY)
