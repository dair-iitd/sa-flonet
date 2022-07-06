import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils

from transformers import BertModel

class SiameseNetWithBert(nn.Module):
    
    def __init__(self, args, glob):
        super(SiameseNetWithBert, self).__init__()
        
        self.margin = 2        
        self.softmax = nn.Softmax(dim=1)
        
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768, 1)
        self.bceloss = nn.BCEWithLogitsLoss()
        
    def forward(self, batch_sentids, batch_segids, batch_mask, labels):
        # bsz * hidden_size
        combined_representation = self.bert_model(batch_sentids, token_type_ids=batch_segids, attention_mask=batch_mask)[1]
        prediction = self.fc1(combined_representation).view(-1)
        loss = self.bceloss(prediction, labels.float())
        distance = 1-prediction

        return loss, distance

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