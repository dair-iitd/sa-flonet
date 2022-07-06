import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import PAD_INDEX, EOU_INDEX

from .ProxyInputEncoder import ProxyInputEncoder

class ProxyNet2(nn.Module):
    
    def __init__(self, args, glob):
        super(ProxyNet2, self).__init__()
        self.margin = args.margin
        self.input_encoder = ProxyInputEncoder(args, glob)
        self.ce_loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)


    def encode_input(self,contexts,context_utterance_lengths,context_lengths):
        encoded_input= self.input_encoder(contexts,context_utterance_lengths,context_lengths)
        return encoded_input

    def forward(self, context, context_utt_lens, context_lens, path, path_utt_lens, path_lens, labels):
        
        context_representation = self.encode_input(context, context_utt_lens, context_lens)#b,dim
        path_representation = self.encode_input(path, path_utt_lens, path_lens)#n_chart_nodes,dim

        #calculate scores
        scores = torch.mm(context_representation, torch.transpose(path_representation,0,1))#b,n_chart_nodes
        loss = self.ce_loss(scores,labels)

        return loss, self.softmax(scores)

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
