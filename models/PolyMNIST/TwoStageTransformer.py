from typing import Dict
import torch.nn as nn
import torch
from omegaconf import OmegaConf
import sys
import copy
from einops import rearrange,repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
sys.path.append('/home/siyi/project/mm/mul_foundation/Dynamic_Missing')
from models.PolyMNIST.MissTransformer import MissTransformer

class UnimodalEncoder(nn.Module):
    def __init__(self,):
        super(UnimodalEncoder, self).__init__()
        self.network  = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),     # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # -> (128, 4, 4)
            nn.ReLU(),                                
        )
        self.proj_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),   # -> (128, 4, 4)
            nn.MaxPool2d(2, 2)                                         # -> (128, 2, 2)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.network(x)
        x = self.proj_conv(x)
        return x


class TwoStageTransformer(nn.Module):
    '''
    Have a frozen imputation network to generate the missing modalities.
    Input is a dictionary of modalities and a mask indication matrix.
    Encode all modalities through modality-specific CNN encoders.
    Use the mask to select existing modalities and pass them through a transformer.
    Still input masked tokens into the transformer, use attention mask to avoid attending to masked tokens.
    Special embeddings: cls token (1,dim), modality embeddings (num_modalities, dim), intra-modality positional embeddings (1+sum_num_patches, dim)
    '''
    def __init__(self, args):
        super(TwoStageTransformer, self).__init__()
        assert args[args.dataset_name].transformer_checkpoint is not None, 'Please provide the transformer checkpoint'
        self.model = MissTransformer(args)

        self.create_imputation_network(args)
    
    def create_imputation_network(self, args):
        assert args[args.dataset_name].imputer_name == 'MoPoE', 'Only MoPoE is supported'
        args_imputer = copy.deepcopy(args)
        args_imputer.checkpoint = args[args.dataset_name].imputer_checkpoint
        from models.PolyMNIST.MoPoE import MoPoE
        self.imputer = MoPoE(args_imputer)
        for param in self.imputer.parameters():
            param.requires_grad = False
        print(f"Imputer {args[args.dataset_name].imputer_name} frozen")


    def forward(self, x: Dict, mask: torch.Tensor):
        with torch.no_grad():
            x = self.imputer(x, mask)

        out = self.model(x, mask=torch.zeros_like(mask).bool())
        return out


if __name__ == "__main__":
    args = {'PolyMNIST':{"transformer_dim": 128, "transformer_heads": 4, "transformer_layers": 2, "transformer_drop": 0.0, 
                         "num_patches_list": [4, 4, 4, 4, 4], "num_masks": 0, 'imputer_name': 'MoPoE',
                         'imputer_checkpoint': '/home/siyi/project/mm/result/Dynamic_project/PM23/MoPoE_MMNIST_2025_04_01_17_18_11_421525/checkpoints/0299/mm_vae',
                         'transformer_checkpoint': '/home/siyi/project/mm/result/Dynamic_project/PM16/DynamicTransformer_singleCLS_singleCLS_whole_none_PolyMNIST_DynamicTransformer_singleCLS_0426_151226/downstream/checkpoint_best_acc.ckpt'}, 
                         'checkpoint': None,
                         'dataset_name': 'PolyMNIST', 'batch_size':2, 'logdir':None,
                "num_modalities": 5}
    args = OmegaConf.create(args)
    model = TwoStageTransformer(args)
    x = {'m0': torch.randn(2, 3, 28, 28), 'm1': torch.randn(2, 3, 28, 28), 'm2': torch.randn(2, 3, 28, 28),
         'm3': torch.randn(2, 3, 28, 28), 'm4': torch.randn(2, 3, 28, 28)}
    mask = torch.tensor([[True, True, False, False, False], [True, True, False, False, False]])
    output = model.forward(x, mask)
    print(output.shape)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

        

        
