import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils
import pickle
from utils import get_embedding_layer#, get_embedding_matrix_from_pretrained
    
class U_Encoder_Transform(nn.Module):
    def __init__(self,args):
        super(U_Encoder_Transform, self).__init__()
        #embedding stuff
        self.emb_lookup = get_embedding_layer(self.load_embedding_matrix(args),non_trainable=False)
        self.speaker_lookup = nn.Embedding(2, args.emb_size)
        self.history_lookup = nn.Embedding(200, args.emb_size)
        self.position_lookup = nn.Embedding(1024, args.emb_size)
        self.LayerNorm = nn.LayerNorm(args.emb_size)
        self.dropout = nn.Dropout(args.dropout)
        self.cuda = args.cuda
        self.register_buffer("position_ids", torch.arange(1024).expand((1, -1)))

        #transformer stuff
        self.n_heads = args.n_heads
        transformer = nn.TransformerEncoderLayer(args.emb_size, args.n_heads, args.encoder_hidden_size, args.dropout)
        self.transformer = nn.TransformerEncoder(transformer, args.encoder_num_layers,norm=nn.LayerNorm(args.emb_size))

    def forward(self, utt, utt_lens, utt_token_type):
        embeddings, key_padding_mask, _ = self.get_input_embedding(utt, utt_lens, utt_token_type)
        embeddings = torch.transpose(embeddings,0,1)#batch is input as second dimension
        output = self.transformer(embeddings,src_key_padding_mask=key_padding_mask)
        return torch.mean(output,dim=0)#use CLS token encoding
    
    def make_attention_mask(self, utt_pos):
        max_len = utt_pos.shape[1]
        lens = torch.sum(utt_pos==False,dim=-1).cpu().numpy()
        n_heads = self.n_heads
        mask_ = torch.zeros(len(lens)*n_heads,max_len,max_len).bool()
        for i,d in enumerate(lens):
            mask = torch.ones(d,d)
            pad = max_len-d
            pad_mask = torch.nn.ZeroPad2d([0,pad,0,pad])
            mask = pad_mask(mask)
            mask = mask==0
            mask_[0+i*n_heads:(i+1)*n_heads]=mask
        return mask_

    def get_input_embedding(self, utt, utt_lens, utt_token_type):
        u_embeds = self.emb_lookup(utt)
        u_s_embeds = self.speaker_lookup(utt_token_type[0])
        u_h_embeds = self.history_lookup(utt_token_type[1])
        u_p_embeds = self.position_lookup(self.position_ids[:,utt.shape[-1]])

        embeddings = u_embeds + u_s_embeds + u_p_embeds + u_h_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        key_padding_mask = utt==0
        attention_mask = self.make_attention_mask( key_padding_mask)
        if self.cuda:
            embeddings, key_padding_mask, attention_mask = embeddings.cuda(), key_padding_mask.cuda(), attention_mask.cuda()

        return embeddings, key_padding_mask, attention_mask

    def load_embedding_matrix(self,args):
        with open(args.saved_glove_path,"rb") as f:
            matrix = pickle.load(f)
            num_embeddings, embedding_dim = matrix.shape
            assert embedding_dim == args.emb_size
            assert num_embeddings == args.encoder_vocab_size
            return matrix