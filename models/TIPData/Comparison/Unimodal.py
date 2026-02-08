'''
For ablation and DynMM models
'''
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import json
import sys
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.utils.TIP_utils.Transformer import TabularTransformerEncoder
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder



class Unimodal(nn.Module):
    '''
    Unimodal model. 
    Input is x and mask
    x: dictionary of modalities, e.g., {'m0': x0, 'm1': x1, ...}
    mask: tensor of shape (B, num_modalities), where True means the modality is missing
    Only support whole missing, i.e., only one modality is non-missing and the rest are missing.
    '''
    def __init__(self, args, proceed_modality_id=None):
        super(Unimodal, self).__init__()
        self.modality_names = args.modality_names
        self.field_lengths_tabular = torch.load(args.DATA_field_lengths_tabular)
        self.cat_lengths_tabular = []
        self.con_lengths_tabular = []
        self.embedding_size = args[args.dataset_name].embedding_size
        for x in self.field_lengths_tabular:
            if x == 1:
                self.con_lengths_tabular.append(x) 
            else:
                self.cat_lengths_tabular.append(x)
        flags = OmegaConf.create({'tabular_embedding_dim': 512, 'embedding_dropout': 0.0,
                                    'tabular_transformer_num_layers': 4, 'multimodal_transformer_num_layers': 4,        
                                'multimodal_embedding_dim': 512, 'drop_rate': 0.0, 'checkpoint': None})
        if proceed_modality_id is not None:
            self.proceed_modality_id = proceed_modality_id
        else:
            self.proceed_modality_id = args[args.dataset_name].proceed_modality_id
        self.proceed_modality_name = self.modality_names[self.proceed_modality_id]
        print(f'Unimodal modality. ID: {self.proceed_modality_id}, name: {self.proceed_modality_name}')
        self.embedding_size = args[args.dataset_name].embedding_size
        if self.proceed_modality_name == 'img':
            self.encoder = torchvision_ssl_encoder('resnet50', pretrained=True, return_all_feature_maps=True)
            self.mapper = nn.Linear(2048, self.embedding_size)
        elif self.proceed_modality_name == 'tabular':
            self.encoder = TabularTransformerEncoder(flags, self.cat_lengths_tabular, self.con_lengths_tabular)
            self.mapper = nn.Linear(512, self.embedding_size)
        else:
            raise ValueError(f'Unknown modality name: {self.proceed_modality_name}')
        self.classifier = nn.Linear(self.embedding_size, args.num_classes)
    
    def forward_feature(self, x: Dict, mask: torch.Tensor):
        out = x[self.proceed_modality_name]
        out = self.encoder(out)
        if self.proceed_modality_name == 'img':
            out = F.adaptive_avg_pool2d(out[-1], (1, 1)).flatten(1)
        elif self.proceed_modality_name == 'tabular':
            out = out[:, 0]
        out = self.mapper(out)
        return out
    
    def forward(self, x: Dict, mask: torch.Tensor):
        x = self.forward_feature(x, mask)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    args = {'DVM':{"embedding_size": 256, 'proceed_modality_id': 0}, 
                          'dataset_name': 'DVM', 'modality_names': ['img', 'tabular'], 'DATA_field_lengths_tabular': '/bigdata/siyi/data/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
                "num_modalities": 2, "checkpoint": None, "num_classes": 283, 'batch_size':2,'missing_tabular': False}
    args = OmegaConf.create(args)
    model = Unimodal(args)
    x = {'img': torch.randn(2, 3, 128, 128), 'tabular': torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                    [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)}
    mask = torch.tensor([[False, False], [False, False]])
    output = model.forward(x, mask)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

