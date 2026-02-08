'''
For ablation and DynMM models
'''
from typing import Dict
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import json
import sys
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.TIPData.Comparison.Unimodal import Unimodal


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

        self.m_encoders = nn.ModuleDict()
        for i, name in enumerate(self.modality_names):
            self.m_encoders[name] = Unimodal(args, proceed_modality_id=i)

        self.use_imputer = args[args.dataset_name].use_imputer
        print(f"Use imputer: {self.use_imputer}")
        self.create_imputation_network(args)

        self.classifier = nn.Linear(self.embedding_size * self.num_modalities, args.num_classes)


    def create_imputation_network(self, args):
        self.imputer_name = args[args.dataset_name].imputer_name
        if self.imputer_name  == 'TIP':
            args_imputer = args
            args_imputer.checkpoint = args[args.dataset_name].imputer_checkpoint
            args_imputer.field_lengths_tabular = args.DATA_field_lengths_tabular
            from models.utils.TIP_utils.BaseReconTIP import ReconTIP
            self.imputer = ReconTIP(args_imputer)
            for param in self.imputer.parameters():
                param.requires_grad = False
        elif self.imputer_name  == 'ML':
            # Input has already been imputed through machine learning methods
            self.imputer = None  
        else:
            raise ValueError(f"Unsupported imputer name: {self.imputer_name}")
        print(f"Imputer {self.imputer_name } frozen")
    

    def forward(self, x: Dict, mask: torch.Tensor):
        # x: name: value; mask: (B, M) 1 means missing
        mask = mask.bool()  
        original_mask = mask.clone()  # save the original mask for imputation
        if self.use_imputer:
            assert mask[0].sum() > 0, 'Mask should have at least one missing modality'
            assert not torch.any(mask[:, 0]), f'TIP does not support missing imaging features, but got {mask[:, 0].sum()} missing values in imaging features.'
            with torch.no_grad():
                if 'tabular_missing_mask' in x:
                    tabular_missing_mask = x['tabular_missing_mask']
                else:
                    tabular_missing_mask = torch.full_like(x['tabular'], fill_value=mask[0, 1], dtype=torch.bool)
                    # tabular_missing_mask = torch.fill_(torch.empty(x['tabular'].shape[0], x['tabular'].shape[1]), mask[0,1]).bool().to(mask.device)
                if self.imputer_name == 'TIP':
                    x = self.imputer(x, tabular_missing_mask)
                elif self.imputer_name == 'ML':
                    pass
            mask = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)  # after imputation, no missing modalities
        else:
            assert torch.sum(mask, dim=(0,1)) == 0, "Only supports non-missing data when use_imputer is False"

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
    args = {'DVM':{"embedding_size": 256, "use_imputer": False,
                   'imputer_name': 'TIP', 'imputer_checkpoint': '/vol/biomedic3/sd1523/project/mm/result/TIP_results/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt',}, 
                         'dataset_name': 'DVM', 'modality_names': ['img', 'tabular'], 'DATA_field_lengths_tabular': '/bigdata/siyi/data/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
                "num_modalities": 2, "checkpoint": None, "num_classes": 286, 'missing_tabular': False}
    args = OmegaConf.create(args)
    model = Multimodal_CONCAT(args)
    x = {'img': torch.randn(2, 3, 128, 128), 'tabular': torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                    [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)}
    mask = torch.tensor([[False, False], [False, False]])
    output = model.forward(x, mask)
    print(output.shape)

    # test with imputer
    args['DVM'].use_imputer = True
    model = Multimodal_CONCAT(args)
    mask = torch.tensor([[False, True], [False, True]])
    output = model.forward(x, mask)

    # test intra-tabular missing
    args.missing_tabular = True
    model = Multimodal_CONCAT(args)
    x = {'img': torch.randn(2, 3, 128, 128), 'tabular': torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                    [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32),
            'tabular_missing_mask': torch.tensor([[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False],
                                                  [True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False]])}
    mask = torch.tensor([[False, True], [False, True]])
    output = model.forward(x, mask)
    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)