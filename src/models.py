import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, params):
        super(EncoderRNN, self).__init__()
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.input_size = params['inventory_size']
        self.bsz = params['batch_size']
        self.dropout_p = params['dropout_p']
        self.recurrence = params['recurrence']
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        if self.recurrence == 'GRU':
          self.recurrent = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        elif self.recurrence == 'LSTM':
          self.recurrent = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        else:
          self.recurrent = nn.RNN(self.embedding_size, self.hidden_size, batch_first=True, nonlinearity='tanh')

    def forward(self, input, hidden, state): #input is batch size x 1
        embedded = self.embedding(input) #batch size x 1 x embedding size
        embedded = self.dropout(embedded)
        output = embedded
        if self.recurrence == 'LSTM':
          output, (hidden, state) = self.recurrent(output, (hidden, state))
        else:
          output, hidden = self.recurrent(output, hidden) #out is batch x 1 x emb_size, hidden is 1 x batch_size x emb_size
          state = 0
        return output, hidden, state
 

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, params):
        super(DecoderRNN, self).__init__()
        self.emb_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.output_size = params['inventory_size']
        self.dropout_p = params['dropout_p']
        self.in_length = params['max_in_length']
        self.out_length = params['max_out_length']
        self.with_attn = params['with_attn']
        self.recurrence = params['recurrence']
        self.attn_type = params['attn_type']

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.in_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        
        if self.recurrence == 'GRU':
          self.recurrent = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        elif self.recurrence == 'LSTM':
          self.recurrent = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        else:
          self.recurrent = nn.RNN(self.hidden_size, self.hidden_size, batch_first=True, nonlinearity='tanh')
        
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, state, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        if self.with_attn == True:
          concatenated = torch.cat((embedded.squeeze(1), hidden.squeeze(0)), 1) #embedded is [bsz x 1 x d_emb],  hidden is [1 x bsz x d_emb], concatenated is [bsz x d_emb * 2]

          if self.attn_type == 'weighted':
            concatenated = torch.cat((embedded.squeeze(1), hidden.squeeze(0)), 1) #embedded is [bsz x 1 x d_emb],  hidden is [1 x bsz x d_emb], concatenated is [bsz x d_emb * 2]
            attended = self.attn(concatenated) #[bsz x seq_len], seq len is num hidden states in encoder
            attn_weights = F.softmax(attended, dim=1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(1,2))
            output = torch.cat((embedded, attn_applied), 2)
            output = self.attn_combine(output)
            
          else:
            attn_weights = torch.bmm(hidden.transpose(0,1), encoder_outputs)
            attn_weights = nn.functional.softmax(attn_weights, 2)
            attn_applied = torch.bmm(attn_weights, encoder_outputs.transpose(1,2))
            output = torch.cat((embedded, attn_applied), 2)
            output = self.attn_combine(output)
          
        else:
            output = embedded
            attn_weights = torch.zeros(self.in_length)

        output = torch.tanh(output)
        if self.recurrence == 'LSTM':
          output, (hidden,state) = self.recurrent(output, (hidden,state))
        else:
          output, hidden = self.recurrent(output, hidden)
          state = 0
        output = self.out(output)
        output = F.log_softmax(output, dim=2)

        return output, hidden, state, attn_weights