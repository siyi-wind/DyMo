import torch
import torch.nn as nn
from omegaconf import OmegaConf
import sys
import json
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.utils.MUSE_utils.gnn import MML
from models.CelebA.networks.ConvNetworksImgCelebA import EncoderImg
from models.CelebA.networks.ConvNetworksTextCelebA import EncoderText


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
        num_text_features = len(args.alphabet)
        self.m_encoders = nn.ModuleDict({
                'img': EncoderImg(None),
                'text': EncoderText(None),
                })
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
    alphabet_path = join(project_path, 'datasets/alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = OmegaConf.create({
        "dataset_name": "CelebA", "CelebA": { "embedding_size": 32,},
        "num_modalities": 2, 'alphabet': alphabet,
        "modality_names": ['img', 'text'],
        'num_classes': 2,})
    model = MUSE(
        args,
        dropout=0.1,
        ffn_layers=2,
        gnn_layers=2,
        gnn_norm=None,
    )

    x = {'img': torch.randn(2, 3, 64, 64), 'text': torch.randn(2, 256, 71)}
    mask = torch.tensor([[True, False], [True, False]])
    label = torch.tensor([1,0])
    loss, logits = model.forward_train(x, mask, label)
    print("Loss:", loss)
    out = model.forward(x, mask)
    print(out)
