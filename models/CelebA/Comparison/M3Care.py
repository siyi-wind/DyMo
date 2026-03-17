from typing import Dict
import torch.nn as nn
import torch
from omegaconf import OmegaConf
import sys
import json
from einops import rearrange,repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.utils.transformer import Block
from models.utils.M3Care_utils.gnn import GNNImputation
from models.CelebA.networks.ConvNetworksImgCelebA import EncoderImg
from models.CelebA.networks.ConvNetworksTextCelebA import EncoderText
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class M3Care(nn.Module):
    '''
    Input is a dictionary of modalities and a mask indication matrix.
    Encode all modalities through modality-specific CNN encoders.
    Use the mask to select existing modalities and pass them through a transformer.
    Still input masked tokens into the transformer, use attention mask to avoid attending to masked tokens.
    Special embeddings: cls token (1,dim), modality embeddings (num_modalities, dim), intra-modality positional embeddings (1+sum_num_patches, dim)
    '''
    def __init__(self, args):
        super(M3Care, self).__init__()
        num_text_features = len(args.alphabet)
        self.m_encoders = nn.ModuleDict({
                'img': EncoderImg(None),
                'text': EncoderText(None),
                })
        dim = args[args.dataset_name].transformer_dim
        num_heads = args[args.dataset_name].transformer_heads
        num_layers = args[args.dataset_name].transformer_layers
        drop = args[args.dataset_name].transformer_drop
        num_patches_list = args[args.dataset_name].num_patches_list
        self.modality_names = args.modality_names
        self.num_masks = args[args.dataset_name].num_masks
        print('Randomly drop {} modalities'.format(self.num_masks))

        self.m_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(args.num_modalities)])
        self.imputer = GNNImputation(args)
        self.transformer = nn.ModuleList([
                            Block(dim=dim, num_heads=num_heads, 
                                  drop=drop, is_cross_attention=False) 
                            for _ in range(num_layers)
                            ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.modality_embeddings = nn.Embedding(args.num_modalities, dim)
        self.num_modalities = args.num_modalities
        self.num_patches_list = num_patches_list # number of patches in each modality
        self.pos_embeddings = nn.Parameter(torch.zeros(1, (1+sum(self.num_patches_list)), dim))
        self.classifier = nn.Linear(dim, args.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lambda_stab = args[args.dataset_name].lambda_stab 
        self.create_subset2id()
        self.K = args[args.dataset_name].augmentation_K 
        print(f'Use {self.K} random subsets for training')

        if not args.checkpoint:
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.pos_embeddings, std=.02)
            self.apply(self._init_weights)
            print('Initialize MissTransformer from scratch')
        else:
            self.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['state_dict'], strict=False)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=.02)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()

    def create_subset2id(self):
        # create a dictionary to map each combination of modalities to an id
        modalities = list(range(self.num_modalities))
        self.subsets = list(powerset(modalities))
        self.subsets = [comb for comb in self.subsets if len(comb) > 0]
        self.subset2id = {}
        self.id2subset = {}
        for i, comb in enumerate(self.subsets):
            mask = torch.ones(len(modalities), dtype=torch.bool)
            mask[list(comb)] = False
            self.subset2id[tuple(mask.tolist())] = i
            self.id2subset[i] = tuple(mask.tolist())
        self.num_subsets = len(self.subsets)

    def forward_logits(self, x: Dict, mask: torch.Tensor):
        assert self.num_modalities == len(x)

        # encoding
        out = {}
        for i, name in enumerate(self.modality_names):
            tmp = self.m_encoders[name](x[name])  # (B, C)
            tmp = self.m_norms[i](tmp)  # (B, C)
            mask_i = mask[:, i]
            out[name] = tmp * ((~(mask_i.unsqueeze(-1))).float())  # (B, C)
        
        # GNN imputation, notice in GNNImputation, mask=1 means non-missing modality
        x, L_stab = self.imputer(out, ~mask)  # Dict

        # modality embedding
        out = []
        for i, name in enumerate(self.modality_names):
            tmp = x[name].unsqueeze(1)  # (B, 1, C)
            mask_i = mask[:, i].unsqueeze(1)  # (B, 1)
            modality_embed = self.modality_embeddings(torch.full_like(mask_i, i).long().to(tmp.device))  # (B, 1, C)
            tmp = tmp + modality_embed
            out.append(tmp)
        
        out = torch.cat(out, dim=1)  # (B, P, C)
            
        B, P, C = out.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        out = torch.cat([cls_tokens, out], dim=1)
        out = out + self.pos_embeddings

        for block in self.transformer:
            out = block(out)
        out = out[:, 0, :]
        out = self.classifier(out)
        return out, L_stab

    def forward_train(self, x: Dict, mask: torch.Tensor, y: torch.Tensor = None):
        if self.K is None:
            logits, L_stab = self.forward_logits(x, mask)
            loss_ce = self.criterion(logits, y)
            loss = loss_ce + L_stab * self.lambda_stab
            return (loss, logits)
        
        loss_list = []
        sample_idx = torch.randperm(len(self.id2subset), device=mask.device)[:self.K]

        for id in sample_idx:
            mask = torch.tensor(self.id2subset[int(id)], device=mask.device)
            mask = mask.unsqueeze(0).expand(len(y), -1)
            logits, L_stab = self.forward_logits(x, mask)
            loss_ce = self.criterion(logits, y)
            loss = loss_ce + L_stab * self.lambda_stab
            loss_list.append(loss)

        loss = sum(loss_list) / len(loss_list)
        return (loss, logits)
    
    def forward(self, x: Dict, mask: torch.Tensor):
        logits, _ = self.forward_logits(x, mask)
        return logits
    


if __name__ == "__main__":
    alphabet_path = join(project_path, 'datasets/alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = {'CelebA':{"transformer_dim": 32, "transformer_heads": 2, "transformer_layers": 2, "transformer_drop": 0.0, 
                         "num_patches_list": [1, 1], 'embedding_size': 32, 'num_gnn_layers': 2, 'lambda_stab': 1e-8,
                         'num_masks': 0, 'augmentation_K': None}, 
                         'dataset_name': 'CelebA', 'num_classes': 2,
                         'alphabet': alphabet, 'modality_names': ['mnist', 'svhn', 'text'],'modality_names': ['img', 'text'],
                "num_modalities": 2, "checkpoint": None}
    args = OmegaConf.create(args)
    model = M3Care(args)
    x = {'img': torch.randn(2, 3, 64, 64), 'text': torch.randn(2, 256, 71)}
    mask = torch.tensor([[True, False], [True, False]])
    loss, out = model.forward_train(x, mask, torch.tensor([1, 0]))
    print(loss)
    print(out.shape)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

        

        
