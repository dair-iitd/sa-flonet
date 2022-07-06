import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.rnn as rnn_utils

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import PAD_INDEX, EOU_INDEX
from .ProxyInputEncoder import ProxyInputEncoder
from .U_Encoder_Transform import U_Encoder_Transform

class ProxyScore(nn.Module):
    
    def __init__(self, args, glob):
        super(ProxyScore, self).__init__()
        self.margin = args.margin
        self.use_transformer = args.use_transformer
        if not self.use_transformer:
            self.input_encoder = ProxyInputEncoder(args, glob)
            self.path_encoder = ProxyInputEncoder(args, glob)
            self.linear = nn.Linear(2*args.encoder_hidden_size,1,bias=False)
        else:
            self.transformer = U_Encoder_Transform(args)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.loss = nn.MSELoss()

    def forward(self, context, context_utt_lens, context_lens, path, path_utt_lens, path_lens, scores):
        if not self.use_transformer:
            predicted_scores = self.get_scores(context, context_utt_lens, context_lens, path, path_utt_lens, path_lens)
        else:
            predicted_scores = self.get_scores_from_transformer(context, context_utt_lens, context_lens, path, path_utt_lens, path_lens)

        loss = self.get_loss(predicted_scores, scores)

        return loss, predicted_scores

    def get_scores(self, context, context_utt_lens, context_lens, path, path_utt_lens, path_lens):
        context_representation = self.input_encoder(context, context_utt_lens, context_lens)
        path_representation = self.input_encoder(path, path_utt_lens, path_lens)
        #predicted_scores = torch.mm(context_representation, torch.transpose(path_representation,0,1))
        #predicted_scores = context_representation + path_representation
        #predicted_scores = self.linear(predicted_scores)
        context_representation_ = context_representation.expand(path_representation.shape[0],context_representation.shape[1])
        euclidean_distance = F.pairwise_distance(context_representation_, path_representation)
        return euclidean_distance#torch.transpose(predicted_scores,0,1)

    def get_scores_from_transformer(self, contexts, batch_context_lengths, batch_context_token_type, paths, batch_path_length, batch_path_token_type):
        context_representation = self.transformer(contexts, batch_context_lengths, batch_context_token_type)
        path_representation = self.transformer(paths, batch_path_length, batch_path_token_type)
        
        ##normalize
        context_norm = torch.norm(context_representation,p=2,dim=1) + 1e-8
        path_norm = torch.norm(path_representation,p=2,dim=1) + 1e-8
        context_representation = context_representation.div(context_norm[:,None])
        path_representation = path_representation.div(path_norm[:,None])
        
        context_representation_ = context_representation.expand(path_representation.shape[0],context_representation.shape[1])
        euclidean_distance = F.pairwise_distance(context_representation_, path_representation)
        return euclidean_distance
        
    def get_loss(self, predicted_scores, scores, eps = 1e-20):
        #scores is target
        #predicted_scores_ = 1-self.softmax(predicted_scores)
        #log_predicted_scores = torch.log(predicted_scores_+eps).float()
        #scores_ = torch.div(scores,torch.sum(scores,axis=-1))
        #loss = - torch.matmul(log_predicted_scores, scores_)
        n=scores.shape[0]
        scores = torch.argsort(scores,dim=-1,descending=True)[:1]
        scores_5_hot = torch.sum(torch.nn.functional.one_hot(scores, n),axis=0).to(predicted_scores)
        loss_contrastive = torch.mean((scores_5_hot) * torch.pow(predicted_scores, 2) +
                                    (1-scores_5_hot) * torch.pow(torch.clamp(self.margin - predicted_scores, min=0.0), 2))
        '''
        predicted_scores = -predicted_scores #invert the polarity, min score = true 
        predicted_scores = self.softmax(predicted_scores)
        log_predicted_scores = torch.log(predicted_scores+eps).float()
        loss = - torch.matmul(log_predicted_scores, scores_5_hot)
        '''
        return loss_contrastive

    def get_combined_loss(self, doc_prob, gpt_ll, eps = 1e-20):
        #scores = scores / torch.mean(scores).detach()
        log_doc_prob = torch.log(doc_prob+eps).float()
        loss = - torch.logsumexp(log_doc_prob + gpt_ll,-1)
        return loss

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
