import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.rnn as rnn_utils

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import PAD_INDEX, EOU_INDEX

from .ProxyInputEncoder import ProxyInputEncoder

class ProxyNet(nn.Module):
    
    def __init__(self, args, glob):
        super(ProxyNet, self).__init__()
        self.margin = args.margin
        self.input_encoder = ProxyInputEncoder(args, glob)

    def encode_input(self,contexts,context_utterance_lengths,context_lengths):
        encoded_input= self.input_encoder(contexts,context_utterance_lengths,context_lengths)
        return encoded_input

    def forward(self, context, context_utt_lens, context_lens, path, path_utt_lens, path_lens, labels):
        
        context_representation = self.encode_input(context, context_utt_lens, context_lens)
        path_representation = self.encode_input(path, path_utt_lens, path_lens)

        # bsz * 1
        euclidean_distance = F.pairwise_distance(context_representation, path_representation)
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
