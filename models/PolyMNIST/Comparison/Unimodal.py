'''
For ablation and DynMM models
'''
from typing import Dict
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import json

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
            nn.Flatten(),   
            nn.Linear(2048, 512),       # modality-shared space                 
        )
    
    def forward(self, x: torch.Tensor):
        x = self.network(x)
        return x


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
        if proceed_modality_id is not None:
            self.proceed_modality_id = proceed_modality_id
        else:
            self.proceed_modality_id = args[args.dataset_name].proceed_modality_id
        self.proceed_modality_name = self.modality_names[self.proceed_modality_id]
        print(f'Unimodal modality. ID: {self.proceed_modality_id}, name: {self.proceed_modality_name}')
        self.embedding_size = args[args.dataset_name].embedding_size
        self.encoder = UnimodalEncoder()
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
    args = {'PolyMNIST':{"embedding_size": 512, 'proceed_modality_id': 0}, 
                         'dataset_name': 'PolyMNIST', 'modality_names': ['m0', 'm1', 'm2', 'm3', 'm4'],
                "num_modalities": 5, "checkpoint": None, "num_classes": 10, }
    args = OmegaConf.create(args)
    model = Unimodal(args)
    x = {'m0': torch.randn(2, 3, 28, 28), 'm1': torch.randn(2, 3, 28, 28), 'm2': torch.randn(2, 3, 28, 28), 'm3': torch.randn(2, 3, 28, 28), 'm4': torch.randn(2, 3, 28, 28)}
    mask = torch.tensor([[False, False, False, False, False], [False, False, False, False, False]])
    output = model.forward(x, mask)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

