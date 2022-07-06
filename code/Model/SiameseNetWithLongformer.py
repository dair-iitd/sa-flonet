import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils

from transformers import LongformerModel

class SiameseNetWithLongformer(nn.Module):
    
    def __init__(self):
        super(SiameseNetWithLongformer, self).__init__()
        
        self.margin = 2        
        self.softmax = nn.Softmax(dim=1)
        
        self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.fc1 = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.fc2 = nn.Linear(self.model.config.hidden_size, 1)

        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        self.bceloss = nn.BCEWithLogitsLoss()
        
    def forward(self, batch_sentids, batch_segids, batch_mask, labels):
        # bsz * hidden_size
        combined_representation = self.model(batch_sentids, attention_mask=batch_mask)
        prediction = self.fc1(combined_representation[0][:,0,:])
        prediction = self.dropout(prediction)
        prediction = self.fc2(prediction).view(-1)
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