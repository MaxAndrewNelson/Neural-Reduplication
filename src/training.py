import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
def batch_predict(inputs, targets, device, enc, dec, ix2phone, final=False):
    if final == True:
        enc.eval()
        dec.eval()

    all_outputs = []

    bsz, input_length = inputs.size()
    encoder_hidden = enc.initHidden(batch_size=bsz)
    encoder_outputs = torch.zeros(bsz, enc.hidden_size, input_length, device=device) #will store the outputs for the encoder at all timepoints
    output_length = 2*input_length+1 #longest possible output, stop criterion if no end symbol is written

    for ei in range(input_length):
        batch_input = inputs[:,ei].unsqueeze(1)
        encoder_output, encoder_hidden = enc.forward(batch_input, encoder_hidden)
        encoder_outputs[:,:,ei] = encoder_output.squeeze(1)

    decoder_input = inputs[:,0].unsqueeze(1)
    decoder_hidden = encoder_hidden 

    for di in range(output_length): 
        decoder_output, decoder_hidden, attn = dec.forward(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1) #just the one most likely index in the output
        decoder_input = topi.detach().squeeze(1)  # detach from history as input

        items = [int(x) for x in decoder_input]
        as_chars = [ix2phone[x] for x in items]
        all_outputs.append(as_chars)

    zipped = list(zip(*all_outputs))
    preds = [''.join(x) for x in zipped]

    num_correct = 0.
    for i in range(len(preds)):
        targ_string = ''.join([ix2phone[x] for x in (targets[i].cpu().clone().numpy())])
        targ_string = targ_string.replace('#', '')[1:-1]
        if '<' + targ_string + '>' == preds[i][:(len(targ_string)+2)]:
            num_correct += 1
            
    prop_correct = str(num_correct/len(inputs))
    
    return(prop_correct)
  
def batch_train(encoder, decoder, device, inputs, targets, devInputs, devTargets, ix2phone, n_epochs, bsz, learning_rate, teacher_forcing, print_every, test_every=50):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)   
    
    print_loss_total = 0

    criterion = nn.NLLLoss()
    num_inputs, input_length = inputs.size()
    num_inputs, output_length = targets.size()

    batches = [(start, start + bsz) for start in range(0, num_inputs, bsz)]

    for i in range(n_epochs):
        ep_loss = 0
        for batch_ix, (start, end) in enumerate(batches):
            batch_loss = 0

            batch_X = inputs[start:end]
            batch_Y = targets[start:end]
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_outputs = torch.zeros(bsz, encoder.hidden_size, input_length, device=device) #will store the outputs for the encoder at all timepoints
            encoder_hidden = encoder.initHidden(batch_size=batch_X.size()[0])

            for ei in range(input_length):
                batch_input = batch_X[:,ei].unsqueeze(1)
                encoder_output, encoder_hidden = encoder.forward(batch_input, encoder_hidden)
                encoder_outputs[:,:,ei] = encoder_hidden.squeeze(1)

            decoder_input = batch_X[:,0].unsqueeze(1)
            decoder_hidden = encoder_hidden 

            for di in range(output_length):
                decoder_output, decoder_hidden, attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1) #just the one most likely index in the output
                if teacher_forcing == True:
                  decoder_input = batch_Y[:,di].unsqueeze(1) #teacher forcing
                else:
                  decoder_input = topi.detach().squeeze(1) #using decode output detach from history as input

                batch_loss += criterion(decoder_output.squeeze(1), batch_Y[:,di])

            ep_loss += batch_loss.detach()
            batch_loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        if i % print_every == 0:
            print("Epoch " + str(i) + " loss: " + str(ep_loss.cpu().clone().numpy()))

        if i % test_every == 0:
            print("Dev accuracy: " + str(batch_predict(devInputs, devTargets, device, encoder, decoder, ix2phone)))