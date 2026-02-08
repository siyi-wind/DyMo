from typing import Dict
import torch.nn as nn
import torch
from omegaconf import OmegaConf
import sys
import json
from einops import rearrange,repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from itertools import chain, combinations
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)

def powerset(iterable):
    "powerset([1,2,3]) --> (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class MoPoE_Classification(nn.Module):
    '''
    Use MoPoE to get a shared hidden representation for all modalities.
    '''
    def __init__(self, args):
        super(MoPoE_Classification, self).__init__()
        self.num_modalities = args.num_modalities
        self.embedding_size = args[args.dataset_name].embedding_size
        self.modality_names = args.modality_names
        self.classifier = nn.Linear(self.embedding_size, args.num_classes)
        self.K = args[args.dataset_name].augmentation_K
        print(f'Using {self.K} augmentations')

        self.create_imputation_network(args)
        self.create_subset2id()
        self.criterion = nn.CrossEntropyLoss()

    def create_subset2id(self):
        # create a dictionary to map each combination of modalities to an id
        modalities = list(range(self.num_modalities))
        self.subsets = list(powerset(modalities))
        self.subsets = [comb for comb in self.subsets if len(comb) > 0]
        self.subset2id = {}
        self.id2subset = {}
        self.subset2name = {}
        for i, comb in enumerate(self.subsets):
            subset_name = '_'.join(self.modality_names[i] for i in sorted(comb))
            mask = torch.ones(len(modalities), dtype=torch.bool)
            mask[list(comb)] = False
            self.subset2id[tuple(mask.tolist())] = i
            self.subset2name[tuple(mask.tolist())] = subset_name
            self.id2subset[i] = tuple(mask.tolist())
        self.num_subsets = len(self.subsets)
    
    def create_imputation_network(self, args):
        assert args[args.dataset_name].imputer_name == 'MoPoE', 'Only MoPoE is supported'
        args_imputer = args
        args_imputer.checkpoint = args[args.dataset_name].imputer_checkpoint
        from models.CelebA.MoPoE import MoPoE
        self.imputer = MoPoE(args_imputer)
        for param in self.imputer.parameters():
            param.requires_grad = False
        print(f"Imputer {args[args.dataset_name].imputer_name} frozen")


    def forward_one_train(self, x: Dict, mask: torch.Tensor):
        with torch.no_grad():
            x = self.imputer.forward_feature(x, mask)

        out = self.classifier(x)
        return out
    
    def forward(self, x: Dict, mask: torch.Tensor):
        return self.forward_one_train(x, mask)
    
    def forward_train(self, x: Dict, mask: torch.Tensor, y: torch.Tensor=None):
        if self.K is None:
            out = self.forward_one_train(x, mask)
            loss = self.criterion(out, y)
            return (loss, out)

        loss_list = []
        sample_idx = torch.randperm(len(self.id2subset), device=mask.device)[:self.K]

        for id in sample_idx:
            mask = torch.tensor(self.id2subset[int(id)], device=mask.device)
            mask = mask.unsqueeze(0).expand(len(y), -1)
            out = self.forward_one_train(x, mask)
            loss = self.criterion(out, y)
            loss_list.append(loss)
        loss = sum(loss_list) / len(loss_list)
        # return loss and output of the last subset
        return (loss, out)
        


if __name__ == "__main__":
    alphabet_path = join(project_path, 'datasets/alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = {'CelebA':{"embedding_size": 32, 'imputer_name': 'MoPoE', 'augmentation_K': 2,
                         'imputer_checkpoint': '/home/siyi/project/mm/result/Dynamic_project/CA11/CelebA/joint_elbo/factorized/laplace_categorical/CelebA_2025_05_15_18_15_43_805671/checkpoints/0199/mm_vae'},
                         'checkpoint': None, 'alphabet': alphabet, 'len_sequence': 256,
                         'dataset_name': 'CelebA', 'batch_size':2, 'logdir':None, 'modality_names': ['img', 'text'],
                "num_modalities": 2, "num_classes": 2}
    args = OmegaConf.create(args)
    model = MoPoE_Classification(args)
    indices = torch.randint(0, 71, (2, 256))
    one_hot_tensor = torch.nn.functional.one_hot(indices, num_classes=71).float()
    x = {'img': torch.rand(2, 3, 64, 64), 'text': one_hot_tensor,}
    mask = torch.tensor([[True, False], [False, True]])
    loss, output = model.forward_train(x, mask, y=torch.tensor([0, 1]))
    print(output.shape)
    print(loss)

    out = model.forward(x, mask)
    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

        

        
