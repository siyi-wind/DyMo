'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2025
'''

from typing import Tuple, List

import torch
from torch import nn
import sys
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../'))
sys.path.append(project_path)
import torch.nn.functional as F

class PrototypeLoss(torch.nn.Module):
    '''
    Push samples to the positive prototype and push away from negative prototypes
    Use label probability to define the loss
    '''
    def __init__(self, temperature, metric_name: 'cosine_similarity') -> None:
        super().__init__()
        self.temperature = temperature
        self.metric_name = metric_name
        print(f'PrototypeLoss metric: {self.metric_name}')
        assert self.metric_name in ['cosine_similarity', 'squared_euclidean'], f'Unsupported metric: {self.metric_name}'

    def forward(self, label: torch.Tensor, prototypes: torch.Tensor, feat: torch.Tensor):
        '''
        label: (B, C) one-hot label
        prototypes: (C, D) prototypes
        feat: (B, D) features
        '''
        if self.metric_name == 'cosine_similarity':
            # calculate similarity between prototypes and features
            # feat = F.normalize(feat, dim=1)
            # prototypes = F.normalize(prototypes, dim=1)
            out = torch.mm(feat, prototypes.t())/self.temperature
        elif self.metric_name == 'squared_euclidean':
            # calculate euclidean distance
            feat = feat.unsqueeze(1)
            prototypes = prototypes.unsqueeze(0)
            out = -((feat - prototypes).pow(2).sum(dim=2)) / self.temperature
        log_out = F.log_softmax(out, dim=1)
        
        loss = -torch.sum(log_out * label, dim=1)
        loss = loss.mean()
        return loss 
    
if __name__ == '__main__':
    label = torch.tensor([[0,1],[1,0],[0,0]])
    prototypes = F.normalize(torch.randn(2, 128))
    feat = F.normalize(torch.randn(3, 128))
    loss_func = PrototypeLoss(1, 'euclidean distance')
    loss = loss_func(label, prototypes, feat)
    print(loss)