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
from models.CelebA.networks.ConvNetworksImgCelebA import EncoderImg
from models.CelebA.networks.ConvNetworksTextCelebA import EncoderText



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
        num_text_features = len(args.alphabet)
        self.modality_names = args.modality_names
        if proceed_modality_id is not None:
            self.proceed_modality_id = proceed_modality_id
        else:
            self.proceed_modality_id = args[args.dataset_name].proceed_modality_id
        self.proceed_modality_name = self.modality_names[self.proceed_modality_id]
        print(f'Unimodal modality. ID: {self.proceed_modality_id}, name: {self.proceed_modality_name}')
        self.embedding_size = args[args.dataset_name].embedding_size
        if self.proceed_modality_name == 'img':
            self.encoder = EncoderImg(None)
        elif self.proceed_modality_name == 'text':
            self.encoder = EncoderText(None)
        else:
            raise ValueError(f'Unknown modality name: {self.proceed_modality_name}')
        self.classifier = nn.Linear(self.embedding_size, args.num_classes)
    
    def forward_feature(self, x: Dict, mask: torch.Tensor):
        out = x[self.proceed_modality_name]
        out = self.encoder(out)
        return out
    
    def forward(self, x: Dict, mask: torch.Tensor):
        x = self.forward_feature(x, mask)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    alphabet_path = join(project_path, 'datasets/alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = {'CelebA':{"embedding_size": 32, 'proceed_modality_id': 1}, 
                         'dataset_name': 'CelebA', 'alphabet': alphabet, 'modality_names': ['img', 'text'],
                "num_modalities": 2, "checkpoint": None, "num_classes": 2, }
    args = OmegaConf.create(args)
    model = Unimodal(args)
    x = {'img': torch.randn(2, 3, 64, 64), 'text': torch.randn(2, 256, 71)}
    mask = torch.tensor([[False, False], [False, False]])
    output = model.forward(x, mask)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

