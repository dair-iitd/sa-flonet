import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.rnn as rnn_utils

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import PAD_INDEX, EOU_INDEX

class SiameseNet(nn.Module):
    
    def __init__(self, args, glob):
        super(SiameseNet, self).__init__()
        
        self.margin = 2        
        self.softmax = nn.Softmax(dim=1)
        
        self.num_layers = 2
        self.embedding_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.encoder_vocab_size = len(glob['encoder_vocab_to_idx'])

        self.emb_lookup = nn.Embedding(self.encoder_vocab_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True, num_layers=self.num_layers)

        self.use_transformer = args.use_transformer
        if(self.use_transformer):
            self.dropout = args.dropout
            self.n_heads = args.attention_heads
            max_input_length = max(glob['max_input_sent_length'], glob['max_node_utterance_length'], glob['max_flowchart_text_length'], glob['max_edge_label_length'])+10
            self.pos_encoder = PositionalEncoding(self.embedding_size, self.dropout, max_len=max_input_length)
            encoder_layers = TransformerEncoderLayer(self.embedding_size, self.n_heads, self.hidden_size, self.dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, self.num_layers)

    def _compute_utt_representation(self, utts, utt_lengths):
        if(self.use_transformer):
            return self._compute_utt_representation_with_transformer(utts, utt_lengths)

        utt_emb = self.emb_lookup(utts)
        utt_emb_packed = rnn_utils.pack_padded_sequence(utt_emb, utt_lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(utt_emb_packed)
        # bidirectional - concat fwd and bwd GRU outputs
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)

        return hidden

    def insert_eou_token(self, utts, utt_lengths):
        batch,seq=utts.shape
        pad, eou = PAD_INDEX, EOU_INDEX
        eou_tokens=torch.ones(batch,1).fill_(eou).to(utts)
        output = torch.zeros(batch,seq+1).fill_(pad)
        mask = torch.zeros(batch,seq+1).float()
        mask[torch.arange(batch),utt_lengths]=1
        output=output+eou_tokens.expand(-1, output.shape[1])*mask
        utts=torch.cat([utts,torch.zeros(batch,1).fill_(pad).to(utts)],dim=1)
        output=output+utts
        return(output.to(utts))

    def _compute_utt_representation_with_transformer(self, utts, utt_lengths):
        #add an end of utterance token at the end of input
        utt_lengths_local = torch.min(utt_lengths,torch.ones(len(utt_lengths)).long()*utts.shape[0])

        utts = self.insert_eou_token(utts, utt_lengths_local)
        utt_emb = self.emb_lookup(utts) * math.sqrt(self.embedding_size) 
        utt_emb = self.pos_encoder(utt_emb)
        mask = torch.arange(utt_emb.shape[1]).expand(len(utt_lengths), utt_emb.shape[1]) > utt_lengths.unsqueeze(1)
        output = self.transformer_encoder(utt_emb.transpose(0,1), src_key_padding_mask= mask)   #transformer input is (seq length, batch size, emb size)
        output = output.transpose(0,1)
        output = output[torch.arange(utt_lengths_local.shape[0]),utt_lengths_local,:]
        return output

    def forward(self, user_utt, user_utt_lens, agent_utt, agent_utt_lens, node_text, node_text_lens, edge_text, edge_text_lens, labels):
        
        ### PAIRWISE
        # in train
        # bsz = 2*no_of_pairwise_examples: one positive and one negative for same dialog
        # in test
        # bsz = no_of_nodes_in_the_flowchart
        
        user_utt_representation = self._compute_utt_representation(user_utt, user_utt_lens)
        agent_utt_representation = self._compute_utt_representation(agent_utt, agent_utt_lens)

        node_text_representation = self._compute_utt_representation(node_text, node_text_lens)
        edge_text_representation = self._compute_utt_representation(edge_text, edge_text_lens)

        # bsz * (2*hidden_size)
        dialog_representation = torch.cat([user_utt_representation, agent_utt_representation], dim=1)
        node_representation = torch.cat([node_text_representation, edge_text_representation], dim=1)

        # bsz * 1
        euclidean_distance = F.pairwise_distance(dialog_representation, node_representation)
        loss_contrastive = torch.mean((labels) * torch.pow(euclidean_distance, 2) +
                                    (1-labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive, euclidean_distance

    def load(self, checkpoint, args):
        self.load_state_dict(checkpoint['state_dict'])
        optimizer = torch.optim.Adam(params=self.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        return optimizer
    
    def save(self, name, optimizer, args):
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'args': args
        }, name)

class PositionalEncoding(nn.Module):

    def __init__(self, n_dims, dropout=0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, n_dims)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_dims, 2).float() * (-math.log(10000.0) / n_dims))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1), :]
        return self.dropout(x)