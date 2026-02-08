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
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.CelebA.Comparison.Unimodal import Unimodal


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
        self.create_imputation_network(args)

        self.m_encoders = nn.ModuleDict()
        for i, name in enumerate(self.modality_names):
            self.m_encoders[name] = Unimodal(args, proceed_modality_id=i)

        self.classifier = nn.Linear(self.embedding_size * self.num_modalities, args.num_classes)


    def create_imputation_network(self, args):
        self.imputer_name = args[args.dataset_name].imputer_name
        if self.imputer_name == 'MoPoE':
            args_imputer = copy.deepcopy(args)
            args_imputer.checkpoint = args[args.dataset_name].imputer_checkpoint
            from models.MST.MoPoE import MoPoE
            self.imputer = MoPoE(args_imputer)
            for param in self.imputer.parameters():
                param.requires_grad = False
        elif self.imputer_name == 'SimpleImputer':
            from models.utils.simple_imputer import SimpleImputer
            self.imputer_p = args[args.dataset_name].imputer_p
            self.imputer = SimpleImputer(modality_names=args.modality_names, p=self.imputer_p, fill_value=0.0)
            print(f'Using Simple Imputer with fill_value 0.0 and p={self.imputer_p}')
        else:
            raise ValueError(f"Unsupported imputer name: {self.imputer_name}")
        print(f"Imputer {args[args.dataset_name].imputer_name} frozen")

    
    def forward(self, x: Dict, mask: torch.Tensor):
        # x: name: value; mask: (B, M) 1 means missing
        original_mask = mask.clone()  # save the original mask for imputation
        if self.use_imputer:
            assert mask[0].sum() > 0, 'Mask should have at least one missing modality'
            with torch.no_grad():
                x = self.imputer(x, mask)
            mask = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)  # after imputation, no missing modalities
        else:
            # assert all mask values should be False, i.e., no missing data
            assert torch.sum(mask, dim=(0,1)) == 0, "Only supports non-missing data if use_imputer is False"
        out = []
        for i, name in enumerate(self.modality_names):
            assert name in x, f'Modality {name} not found in input x'
            assert x[name].shape[0] == mask.shape[0], f'Batch size of modality {name} does not match mask batch size'
            tmp = self.m_encoders[name].forward_feature(x, mask)  # (B, C)
            out.append(tmp)
        out = torch.cat(out, dim=1)
        out = self.classifier(out)  # (B, num_classes)
        return out


if __name__ == "__main__":
    alphabet_path = join(project_path, 'datasets/alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = {'CelebA':{"embedding_size": 32, "use_imputer": False,
                   'imputer_name': 'MoPoE', 'imputer_checkpoint': '/home/siyi/project/mm/result/Dynamic_project/CA11/CelebA/joint_elbo/factorized/laplace_categorical/CelebA_2025_05_15_18_15_43_805671/checkpoints/0199/mm_vae'}, 
                         'dataset_name': 'CelebA', 'alphabet': alphabet, 'modality_names': ['img', 'text'],
                "num_modalities": 2, "checkpoint": None, "num_classes": 2, 'batch_size':2, 'logdir':None,"len_sequence": 256, }
    args = OmegaConf.create(args)
    model = Multimodal_CONCAT(args)
    x = {'img': torch.randn(2, 3, 64, 64), 'text': torch.randn(2, 256, 71)}
    mask = torch.tensor([[False, False], [False, False]])
    output = model.forward(x, mask)
    print(output.shape)

    # test the imputer
    args.CelebA.use_imputer = True
    model = Multimodal_CONCAT(args)
    x = {'img': torch.randn(2, 3, 64, 64), 'text': torch.randn(2, 256, 71)}
    mask = torch.tensor([[True, False], [True, False]])
    loss, output = model.forward(x, mask)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)