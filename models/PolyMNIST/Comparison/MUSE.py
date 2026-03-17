import torch
import torch.nn as nn
from omegaconf import OmegaConf
import sys
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.utils.MUSE_utils.gnn import MML

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
            nn.Flatten(),                                         # -> (128 * 4 * 4)                         
        )
        self.proj_conv = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),                     # -> (128)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.network(x)
        x = self.proj_conv(x)
        return x


class MUSE(nn.Module):
    def __init__(
            self,
            args,
            dropout=0.25,
            ffn_layers=2,
            gnn_layers=2,
            gnn_norm=None,
            device="cpu",
    ):
        super(MUSE, self).__init__()
        self.num_modalities = args.num_modalities
        self.modality_names = args.modality_names
        embedding_size = args[args.dataset_name].embedding_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.ffn_layers = ffn_layers
        self.gnn_layers = gnn_layers
        self.gnn_norm = gnn_norm
        self.device = device

        self.dropout_layer = nn.Dropout(dropout)

        self.m_encoders = nn.ModuleDict()
        for i in range(args.num_modalities):
            self.m_encoders[f'm{i}'] = UnimodalEncoder()
        self.m_mappers = nn.ModuleList([nn.Linear(embedding_size, embedding_size) for _ in range(args.num_modalities)])

        self.mml = MML(num_modalities=args.num_modalities,
                       hidden_channels=embedding_size,
                       num_layers=gnn_layers,
                       dropout=dropout,
                       normalize_embs=gnn_norm,
                       num_classes=args.num_classes,)

    def forward_train(
            self, x, mask, label,
            **kwargs,):
        out  = []
        for i, name in enumerate(self.modality_names):
            mask_i = mask[:,i]
            tmp = self.m_encoders[name](x[name]) 
            tmp = self.m_mappers[i](tmp)
            tmp = tmp * (~(mask_i.unsqueeze(-1))).float()
            out.append(tmp)
        out = torch.stack(out, dim=1)  # (B, M, D)
        x_flag = ~mask   # Notice mask=1 means missing
        label_flag = torch.ones_like(label, dtype=torch.bool, device=label.device)

        # gnn
        loss, logits = self.mml(
            out, x_flag,
            label, label_flag,
        )
        return (loss, logits)

    def forward(
            self, x, mask, 
            **kwargs,):
        out  = []
        for i, name in enumerate(self.modality_names):
            mask_i = mask[:,i]
            tmp = self.m_encoders[name](x[name]) 
            tmp = self.m_mappers[i](tmp)
            tmp = tmp * (~(mask_i.unsqueeze(-1))).float()
            out.append(tmp)
        out = torch.stack(out, dim=1)  # (B, M, D)
        x_flag = ~mask   # Notice mask=1 means missing

        # gnn
        y_scores, logits = self.mml.inference(
            out, x_flag,
        )
        return logits


if __name__ == "__main__":
    args = OmegaConf.create({
        "dataset_name": "PolyMNIST",
        "PolyMNIST": {
            "embedding_size": 128,
        },
        "num_modalities": 5,
        "modality_names": ["m0", "m1", "m2"],
        'num_classes': 10,
    })
    model = MUSE(
        args,
        dropout=0.1,
        ffn_layers=2,
        gnn_layers=2,
        gnn_norm=None,
    )

    x = {'m0': torch.randn(1, 3, 28, 28), 'm1': torch.randn(1, 3, 28, 28), 'm2': torch.randn(1, 3, 28, 28)}
    mask = torch.tensor([[True, True, False]])
    label = torch.tensor([1])
    loss, logits = model.forward_train(x, mask, label)
    print("Loss:", loss)
    out = model.forward(x, mask)
    print(out)
