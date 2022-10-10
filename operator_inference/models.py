import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_length, dropout, bpe, output_size=None, weights=None):
        super(AttentionModel, self).__init__()        
        self.bpe = bpe
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        if weights is not None:
            self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        if output_size is not None:
            self.label = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.Sigmoid()
                )
        self.dropout = nn.Dropout(p=dropout)
        #self.attn_fc_layer = nn.Linear()
        
    def attention_net(self, lstm_output, final_state):

        """ 
        Arguments
        ---------
        
        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM
        
        ---------
        
        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.
                  
        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)
                      
        """
        
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        
        return new_hidden_state
    
    def forward(self, input_sentences):
    
        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        
        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)
        
        """
        batch_size = input_sentences.size(0)
        # (bs, seq_len, word_len)
        input = self.word_embeddings(input_sentences)
        # (bs, seq_len, word_len, embed_dim)
        # print('input: ', input.shape)
        if self.bpe:
            input = input.sum(2)
        input = input.permute(1, 0, 2)

        # h_0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        # c_0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
            
        input = self.dropout(input)
        # output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0)) # final_hidden_state.size() = (1, batch_size, hidden_size) 
        output, (final_hidden_state, final_cell_state) = self.lstm(input) # final_hidden_state.size() = (1, batch_size, hidden_size) 
        
        output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
        
        attn_output = self.attention_net(output, final_hidden_state)
        if self.output_size is None:
            return attn_output
        logits = self.label(attn_output)
        return logits

