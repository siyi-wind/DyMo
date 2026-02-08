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
from models.PolyMNIST.Comparison.Multimodal_CONCAT import Multimodal_CONCAT
from models.PolyMNIST.Comparison.Unimodal import Unimodal
from models.utils.DynMM_utils.utils import DiffSoftmax


class DynMM(nn.Module):
    '''
    Modality-level dynamic multimodal fusion (DynMM) model. https://github.com/zihuixue/DynMM/tree/main
    '''
    def __init__(self, args):
        super(DynMM, self).__init__()
        self.modality_names = args.modality_names
        self.num_modalities = args.num_modalities
        self.embedding_size = args[args.dataset_name].embedding_size
        assert self.num_modalities == len(self.modality_names), f'num_modalities {self.num_modalities} does not match the length of modality_names {len(self.modality_names)}'

        self.use_imputer = args[args.dataset_name].use_imputer
        print(f"Use imputer: {self.use_imputer}")
        if self.use_imputer:
            self.create_imputation_network(args)
        self.create_unimodal_models(args)
        multimodal_args = copy.deepcopy(args)
        multimodal_args[multimodal_args.dataset_name].use_imputer = False  # multimodal model does not use internal imputer
        self.create_multimodal_model(multimodal_args)

        # gating network
        self.num_branches = self.num_modalities + 1 
        self.gate = nn.Sequential(
            nn.Linear(self.embedding_size * self.num_modalities, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.num_branches),
        )
        self.hard_gate = True
        self.temp = 1.0
        self.ce_loss = nn.CrossEntropyLoss()
        self.rate_reg = 0.1

    def create_imputation_network(self, args):
        assert args[args.dataset_name].imputer_name == 'MoPoE', 'Only MoPoE is supported'
        args_imputer = copy.deepcopy(args)
        args_imputer.checkpoint = args[args.dataset_name].imputer_checkpoint
        from models.PolyMNIST.MoPoE import MoPoE
        self.imputer = MoPoE(args_imputer)
        for param in self.imputer.parameters():
            param.requires_grad = False
        print(f"Imputer {args[args.dataset_name].imputer_name} frozen")


    def create_unimodal_models(self, args):
        self.m_unimodal_models = nn.ModuleDict()
        for i, name in enumerate(self.modality_names):
            self.m_unimodal_models[name] = Unimodal(args, proceed_modality_id=i)

        checkpoint = args[args.dataset_name].unimodal_checkpoint
        if checkpoint:
            # checkpoint is a json file storing a dictionary. keys are modality names and values are the paths to the checkpoints
            with open(checkpoint, 'r') as f:
                checkpoint = json.load(f)
            for i, name in enumerate(self.modality_names):
                ckpt_path = checkpoint[name]
                assert ckpt_path is not None, f'Checkpoint for modality {name} is None'
                ckpt = torch.load(ckpt_path, map_location='cpu')
                self.m_unimodal_models[name].load_state_dict(ckpt['state_dict'], strict=True)
                # freeze the unimodal encoders
                for param in self.m_unimodal_models[name].parameters():
                    param.requires_grad = False
            print(f'Load unimodal checkpoints from {checkpoint}')
            print(f'Unimodal encoders are frozen')


    def create_multimodal_model(self, args):
        self.multimodal_model = Multimodal_CONCAT(args)
        checkpoint = args[args.dataset_name].multimodal_checkpoint
        if checkpoint:
            # multimodal checkpoint is a pth file
            ckpt = torch.load(checkpoint, map_location='cpu')
            load_info = self.multimodal_model.load_state_dict(ckpt['state_dict'], strict=False)
            print(f'Load multimodal checkpoint from {checkpoint}')
            print(f" missing keys: {load_info.missing_keys}, unexpected keys: {load_info.unexpected_keys}")
            # unexpected keys should be 0
            assert len(load_info.unexpected_keys) == 0
            # freeze the multimodal model
            for param in self.multimodal_model.parameters():
                param.requires_grad = False
            print(f'Multimodal model is frozen')
            

    def forward_output(self, x: Dict, mask: torch.Tensor, y: torch.Tensor=None, visualize=False):
        original_mask = mask.clone()  # save the original mask for imputation
        if self.use_imputer:
            assert mask[0].sum() > 0, 'Mask should have at least one missing modality'
            with torch.no_grad():
                x = self.imputer(x, mask)
            mask = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)  # after imputation, no missing modalities
        
        # get unimodal features
        out = []
        predictions = []
        for i, name in enumerate(self.modality_names):
            tmp = self.m_unimodal_models[name].forward_feature(x, mask)  # (B, C)
            out.append(tmp)
            tmp = self.m_unimodal_models[name].classifier(tmp)  # (B, num_classes)
            predictions.append(tmp)  # (B, num_classes)
        out = torch.cat(out, dim=1)  # (B, M*C)
        weight = DiffSoftmax(self.gate(out), tau=self.temp, hard=self.hard_gate)  # (B, M+1)

        predictions.append(self.multimodal_model.forward(x, mask))  # (B, num_classes)
        predictions = torch.stack(predictions, dim=1)  # (B, M+1, num_classes)

        # apply the weights to the predictions
        out = predictions * weight.unsqueeze(2)  # (B, M+1, num_classes)
        out = out.sum(dim=1)  # (B, num_classes)
        return out, weight
    

    def forward_train(self, x: Dict, mask: torch.Tensor, y: torch.Tensor):
        out, weight = self.forward_output(x, mask, y)
        # calculate loss
        ce_loss = self.ce_loss(out, y)
        reg_loss = weight[:, -1].mean()  # regularization loss for the multimodal branch
        loss = ce_loss + reg_loss * self.rate_reg
        return (loss, out)
    

    def forward(self, x: Dict, mask: torch.Tensor, visualize=False):
        out, weight = self.forward_output(x, mask, visualize)
        if visualize:
            return out, weight.squeeze(-1)
        else:
            return out
    

if __name__ == "__main__":
    args = {'PolyMNIST':{"embedding_size": 512, 'use_imputer': False,
                         'unimodal_checkpoint':'/home/siyi/project/mm/result/Dynamic_project/PM38/unimodal_checkpoint.json',
                         'multimodal_checkpoint': '/home/siyi/project/mm/result/Dynamic_project/PM38/whole_none_PolyMNIST_Multimodal_CONCAT_0625_180209/downstream/checkpoint_best_acc.ckpt',
                         'imputer_name': 'MoPoE', 'imputer_checkpoint': '/home/siyi/project/mm/result/Dynamic_project/PM23/MoPoE_MMNIST_2025_04_01_17_18_11_421525/checkpoints/0299/mm_vae'}, 
                         'dataset_name': 'PolyMNIST', 'num_classes': 10, 'modality_names': ['m0', 'm1', 'm2', 'm3', 'm4'],
                "num_modalities": 5, "checkpoint": None, "num_classes": 10, 'batch_size':2, 'logdir':None,}
    args = OmegaConf.create(args)
    model = DynMM(args)
    x = {'m0': torch.randn(2, 3, 28, 28), 'm1': torch.randn(2, 3, 28, 28), 'm2': torch.randn(2, 3, 28, 28), 'm3': torch.randn(2, 3, 28, 28), 'm4': torch.randn(2, 3, 28, 28)}
    mask = torch.tensor([[False, False, False, False, False], [False, False, False, False, False]])
    loss, output = model.forward_train(x, mask, y=torch.tensor([0, 1]))
    print(loss)
    print(output.shape)

    args[args.dataset_name].use_imputer = True
    model = DynMM(args)
    mask = torch.tensor([[False, True, False, False, False], [False, False, True, True, False]])
    output, weight = model.forward(x, mask, visualize=True)
    print(output.shape)
    print(weight.shape)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)