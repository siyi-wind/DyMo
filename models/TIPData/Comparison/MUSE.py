import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import sys
import json
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.utils.MUSE_utils.gnn import MML
from models.utils.TIP_utils.Transformer import TabularTransformerEncoder
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder


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
        self.field_lengths_tabular = torch.load(args.DATA_field_lengths_tabular)
        self.cat_lengths_tabular = []
        self.con_lengths_tabular = []
        for x in self.field_lengths_tabular:
            if x == 1:
                self.con_lengths_tabular.append(x) 
            else:
                self.cat_lengths_tabular.append(x)
        flags = OmegaConf.create({'tabular_embedding_dim': 512, 'embedding_dropout': 0.0,
                                    'tabular_transformer_num_layers': 4, 'multimodal_transformer_num_layers': 4,        
                                'multimodal_embedding_dim': 512, 'drop_rate': 0.0, 'checkpoint': None})
        self.m_encoders = nn.ModuleDict({
                'img': torchvision_ssl_encoder('resnet50', pretrained=True, return_all_feature_maps=True),
                'tabular': TabularTransformerEncoder(flags, self.cat_lengths_tabular, self.con_lengths_tabular)
                })
        self.m_mappers = nn.ModuleDict({
                'img': nn.Linear(2048, embedding_size),
                'tabular': nn.Linear(512, embedding_size)
                }) 

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
            if name == 'img':
                tmp = F.adaptive_avg_pool2d(tmp[-1], (1, 1)).flatten(1)
            elif name == 'tabular':
                tmp = tmp[:, 0]
            tmp = self.m_mappers[name](tmp)
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
            if name == 'img':
                tmp = F.adaptive_avg_pool2d(tmp[-1], (1, 1)).flatten(1)
            elif name == 'tabular':
                tmp = tmp[:, 0]
            tmp = self.m_mappers[name](tmp)
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
    alphabet_path = '/home/siyi/project/mm/mul_foundation/MoPoE/alphabet.json'
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = OmegaConf.create({
        "dataset_name": "DVM", "DVM": { "embedding_size": 256,},
        "num_modalities": 2, 
        "modality_names": ['img', 'tabular'], 'DATA_field_lengths_tabular': '/bigdata/siyi/data/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
        'num_classes': 286,})
    model = MUSE(
        args,
        dropout=0.1,
        ffn_layers=2,
        gnn_layers=2,
        gnn_norm=None,
    )

    x = {'img': torch.randn(2, 3, 128, 128), 'tabular': torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                    [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)}
    mask = torch.tensor([[True, False], [True, False]])
    label = torch.tensor([1,0])
    loss, logits = model.forward_train(x, mask, label)
    print("Loss:", loss)
    out = model.forward(x, mask)
    print(out)
