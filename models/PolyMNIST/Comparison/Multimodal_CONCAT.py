'''
For ablation and DynMM models
'''
from typing import Dict
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import json
import sys
import copy
import time
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.PolyMNIST.Comparison.Unimodal import Unimodal


class Multimodal_CONCAT(nn.Module):
    '''
    Multimodal CONCAT model. Only support non-missing data.
    '''
    def __init__(self, args):
        super(Multimodal_CONCAT, self).__init__()
        self.modality_names = args.modality_names
        self.num_modalities = args.num_modalities
        assert self.num_modalities == len(self.modality_names), f'num_modalities {self.num_modalities} does not match the length of modality_names {len(self.modality_names)}'
        self.embedding_size = args[args.dataset_name].embedding_size

        self.use_imputer = args[args.dataset_name].use_imputer
        print(f"Use imputer: {self.use_imputer}")
        if self.use_imputer:
            self.create_imputation_network(args)

        self.m_encoders = nn.ModuleDict()
        for i, name in enumerate(self.modality_names):
            self.m_encoders[name] = Unimodal(args, proceed_modality_id=i)

        self.classifier = nn.Linear(self.embedding_size * self.num_modalities, args.num_classes)

    def create_imputation_network(self, args):
        assert args[args.dataset_name].imputer_name == 'MoPoE', 'Only MoPoE is supported'
        args_imputer = copy.deepcopy(args)
        args_imputer.checkpoint = args[args.dataset_name].imputer_checkpoint
        from models.PolyMNIST.MoPoE import MoPoE
        self.imputer = MoPoE(args_imputer)
        for param in self.imputer.parameters():
            param.requires_grad = False
        print(f"Imputer {args[args.dataset_name].imputer_name} frozen")
    
    def forward(self, x: Dict, mask: torch.Tensor, visualize=False):
        # x: name: value; mask: (B, M) 1 means missing
        # assert all mask values should be False, i.e., no missing data
        if self.use_imputer:
            assert mask[0].sum() > 0, 'Mask should have at least one missing modality'
            with torch.no_grad():
                x = self.imputer(x, mask)
            mask = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)  # after imputation, no missing modalities
        else:
            assert mask.sum() == 0, 'Mask should have no missing modalities when use_imputer is False'

        start_time = time.time()
        out = []
        for i, name in enumerate(self.modality_names):
            assert name in x, f'Modality {name} not found in input x'
            assert x[name].shape[0] == mask.shape[0], f'Batch size of modality {name} does not match mask batch size'
            tmp = self.m_encoders[name].forward_feature(x, mask)  # (B, C)
            out.append(tmp)
        out = torch.cat(out, dim=1)
        out = self.classifier(out)  # (B, num_classes)
        end_time = time.time()
        B = out.shape[0]
        # report inference latency using ms
        inference_time = (end_time - start_time)*1000
        print(f'Inference time for batch size {B}: {inference_time:.4f} ms, per sample: {inference_time/B:.4f} ms')
        return out


if __name__ == "__main__":
    args = {'PolyMNIST':{"embedding_size": 512, "use_imputer": False,
                         'imputer_name': 'MoPoE', 'imputer_checkpoint': '/home/siyi/project/mm/result/Dynamic_project/PM23/MoPoE_MMNIST_2025_04_01_17_18_11_421525/checkpoints/0299/mm_vae'}, 
                         'dataset_name': 'PolyMNIST', 'num_classes': 10, 'modality_names': ['m0', 'm1', 'm2', 'm3', 'm4'],
                "num_modalities": 5, "checkpoint": None, "num_classes": 10, 'batch_size':2, 'logdir':None,}
    args = OmegaConf.create(args)
    model = Multimodal_CONCAT(args)
    x = {'m0': torch.randn(2, 3, 28, 28), 'm1': torch.randn(2, 3, 28, 28), 'm2': torch.randn(2, 3, 28, 28), 'm3': torch.randn(2, 3, 28, 28), 'm4': torch.randn(2, 3, 28, 28)}
    mask = torch.tensor([[False, False, False, False, False], [False, False, False, False, False]])
    output = model.forward(x, mask)
    print(output.shape)

    # test missing modalities
    args[args.dataset_name].use_imputer = True
    model = Multimodal_CONCAT(args)
    mask = torch.tensor([[True, True, False, False, False], [True, True, False, False, False]])
    output = model.forward(x, mask)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)