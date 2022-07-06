import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .Input_Encoder import Input_Encoder
from .BeamSearch import BeamSearch
from .Decoder import Decoder
from copy import deepcopy
from utils import EOS_INDEX,special_tokens_ids#, get_embedding_layer, get_embedding_matrix_from_pretrained

class MemoryN2N(nn.Module):
    
    def __init__(self, args, glob):
        super(MemoryN2N, self).__init__()
        self.story_vocab_size = args.story_vocab_size
        self.embedding_dimension = args.emb_size
        self.hops = args.hops
        self.encoder_bidirectional_flag = args.encoder_bidirectional_flag==True
        self.encoder_hidden_size = args.encoder_hidden_size
        self.encoder_num_layers = args.encoder_num_layers
        self.loss = args.loss_type
        self.beam_size = args.beam_size

        self.A = nn.Embedding(self.story_vocab_size, self.embedding_dimension)#get_embedding_layer(get_embedding_matrix_from_pretrained(glob, args))
        #self.B = nn.Embedding(self.encoder_vocab_size, self.embedding_dimension)#if RNN like
        self.C = nn.Embedding(self.story_vocab_size, self.embedding_dimension)#get_embedding_layer(get_embedding_matrix_from_pretrained(glob, args))

        args.hidden_size = self.encoder_hidden_size
        self.input_encoder = Input_Encoder(args, glob)

        self.gru = nn.GRU(self.embedding_dimension, self.encoder_hidden_size, batch_first=True, bidirectional=self.encoder_bidirectional_flag, num_layers=self.encoder_num_layers)

        self.decoder = Decoder(args, glob)

    def encode_memories(self,encoder,memories):
         #encoded_memories is (batch_size,num_stories,sentence_size,embedding)
        encoded_memories = encoder(torch.tensor(memories))
        encoded_memories_shape = encoded_memories.shape
        encoded_memories=encoded_memories.reshape(-1,*encoded_memories.shape[2:])
        #encoded_memories is (batch_size*num_stories,sentence_size, Directions*hidden_size)
        encoded_memories, _ = self.gru(encoded_memories)
        encoded_memories = encoded_memories[:,-1,:]#shape (batch_size*num_stories, Directions*hidden_size)
        encoded_memories = encoded_memories.reshape(*encoded_memories_shape[:2],-1)#shape (batch_size, num_stories,Directions*hidden_size)

        return encoded_memories

    def encode_input(self,contexts,context_utterance_lengths,context_lengths,\
        queries,query_lengths):
        encoded_input= self.input_encoder(contexts,context_utterance_lengths,context_lengths,\
        queries,query_lengths)
        '''
        #ORIGINAL
        #(batch_size,sentence_size,embedding)
        encoded_queries = self.B(torch.tensor(queries))
        #(batch_size,sentence_size,Directions*hidden_size)
        encoded_queries, _ = self.gru(encoded_queries)
        encoded_queries = torch.sum(encoded_queries,1)#shape (batch_size, Directions*hidden_size)
        '''
        return encoded_input

    def compute_loss(self, output, responses, response_lengths):
        responses_new = torch.zeros(responses.shape).to(responses).long()-100
        for i,length in enumerate(response_lengths):
            responses_new[i, :length] = responses[i,:length]
        responses = deepcopy(responses_new)
        batch_size, sentence_size, vocab_size = output.shape
        output = output.view(-1, vocab_size)
        responses = responses.view(-1)
        assert output.shape[0] == responses.shape[0]

        #if self.loss == 'cross_entropy':
        loss = torch.exp(F.cross_entropy(output, responses.clone().detach(), ignore_index=-100, reduction='mean'))
        return loss

    def encode(self,contexts,context_utterance_lengths,context_lengths,\
        queries,query_lengths,memories):
        input_memory = self.encode_memories(self.A,memories)
        output_memory = self.encode_memories(self.C,memories)
        queries = self.encode_input(contexts,context_utterance_lengths,context_lengths,\
        queries,query_lengths)
        queries = queries.unsqueeze(-1)

        for _ in range(self.hops):
            probabilities = torch.matmul(input_memory, queries)
            probabilities = nn.Softmax(dim=1)(probabilities) #(batch_size,num_stories,1)

            probabilities = probabilities.permute(0,2,1)
            output = torch.matmul(probabilities,output_memory) #(batch_size,1,2*hidden)
            output = output.permute(0,2,1)
            queries = output + queries

        queries = queries.squeeze(-1) #(batch_size,2*hidden)
        return queries

    def forward(self,contexts,context_utterance_lengths,context_lengths,\
        queries,query_lengths,responses,response_lengths,memories):
        queries = self.encode(contexts,context_utterance_lengths,context_lengths,\
        queries,query_lengths,memories)

        #decode using queries as context
        output=self.decoder(queries,responses,response_lengths)#(batch,max_len,embedding)
        loss = self.compute_loss(output,responses,response_lengths)

        return loss, output

    def position_encoding(self, sentence_size, embedding_size):
        """
        Position Encoding described in section 4.1 [1]
        """
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size+1
        le = embedding_size+1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        # Make position encoding of time words identity to avoid modifying them 
        #encoding[:, -1] = 1.0

        encoding = np.transpose(encoding)
        encoding = torch.tensor(encoding, requires_grad=False)
        return encoding
    
    def position_encoder(self, memories):
        #(batch_size,num_stories,sentence_size,embedding)
        input_memory = self.A(memories) * self.positional_embedding_matrix
        input_memory = input_memory.sum(2)#(batch_size,num_stories,embedding)
        return input_memory

    def top_filtering(self, logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p.
                threshold: a minimal threshold to keep logits
        """
        assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate the code a bit
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

        return logits

    def sample_output(self,contexts,context_utterance_lengths,context_lengths,\
        queries,query_lengths,memories,args):
        #assumes batch size = 1
        queries = self.encode(contexts,context_utterance_lengths,context_lengths,\
        queries,query_lengths,memories)

        output = []
        #beamsearch = BeamSearch(self.decoder,EOS_INDEX)
        #output = beamsearch.get_samples(queries,self.beam_size)
        responses = torch.zeros(1,0).to(args.device).long()
        response_lengths = torch.tensor([0]).to(args.device).long()
        for i in range(args.max_length):
            logits = self.decoder.sample_step(queries,responses,response_lengths)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[0, -1, :] / args.temperature
            logits = self.top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
            if i < args.min_length and prev.item() in special_tokens_ids:
                n_iter = 0
                while prev.item() in special_tokens_ids and n_iter < args.sample_turns:
                    n_iter+=1
                    if probs.max().item() == 1:
                        print("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() in special_tokens_ids:
                break
            output.append(prev.item())
            responses = torch.tensor(output).unsqueeze(0).long().to(args.device)
            response_lengths+=1
            response_lengths=response_lengths.long().to(args.device)
        return output

    def save(self, name, optimizer, args):
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'args': args
        }, name)