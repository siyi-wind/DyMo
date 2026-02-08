from typing import Dict
import torch.nn as nn
import torch
from omegaconf import OmegaConf


class UnimodalModel(nn.Module):
    def __init__(self, args):
        super(UnimodalModel, self).__init__()
        # only support whole missing
        assert 'whole' in args.missing_train and 'whole' in args.missing_val and 'whole' in args.missing_test
        self.network  = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),     # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # -> (128, 4, 4)
            nn.ReLU(),
            nn.Flatten(),                                                # -> (2048)
            nn.Linear(2048, 512),       # -> (ndim_private + ndim_shared)
            nn.ReLU(),
            nn.Linear(512, 10)            # -> (10)
        )
    
    def forward(self, x: Dict, mask: torch.Tensor):
        mask = mask[0]
        assert torch.sum(~mask) == 1
        id = torch.argmin(mask.float())
        assert mask[id] == False

        x = self.network(x[f'm{id}'])
        return x



if __name__ == "__main__":
    args = {'missing_train': 'whole', 'missing_val': 'whole', 'missing_test': 'whole'}
    args = OmegaConf.create(args)
    model = UnimodalModel(args)
    x = {'m0': torch.randn(1, 3, 28, 28), 'm1': torch.randn(1, 3, 28, 28), 'm2': torch.randn(1, 3, 28, 28)}
    mask = torch.tensor([[False, True, True]])
    output = model(x, mask)
    print(output.shape)