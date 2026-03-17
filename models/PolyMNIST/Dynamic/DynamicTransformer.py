from typing import Dict
import torch.nn as nn
import torch
from omegaconf import OmegaConf
import sys
from einops import rearrange,repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.utils.transformer import Block
from itertools import chain, combinations
import torch.nn.functional as F
import time

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


def powerset(iterable):
    "powerset([1,2,3]) --> (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class DynamicTransformer(nn.Module):
    '''
    Input is a dictionary of modalities and a mask indication matrix.
    Encode all modalities through modality-specific CNN encoders.
    Use the mask to select existing modalities and pass them through a transformer.
    Still input masked tokens into the transformer, use attention mask to avoid attending to masked tokens.
    Special embeddings: cls token (1,dim), modality embeddings (num_modalities, dim), intra-modality positional embeddings (1+sum_num_patches, dim)
    '''
    def __init__(self, args):
        super(DynamicTransformer, self).__init__()
        print('DynamicTransformer')
        self.m_encoders = nn.ModuleList([UnimodalEncoder() for _ in range(args.num_modalities)])
        dim = args[args.dataset_name].transformer_dim
        num_heads = args[args.dataset_name].transformer_heads
        num_layers = args[args.dataset_name].transformer_layers
        drop = args[args.dataset_name].transformer_drop
        num_patches_list = args[args.dataset_name].num_patches_list
        self.num_masks = args[args.dataset_name].num_masks
        self.num_modalities = args.num_modalities
        print('Randomly drop {} modalities'.format(self.num_masks))
        if 'distance_metric' in args[args.dataset_name]:
            self.distance_metric = args[args.dataset_name].distance_metric
        else:
            self.distance_metric = 'cosine_similarity'
        print(f'Use distance metric for DynamicTransformer: {self.distance_metric}')

        self.m_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.num_modalities)])
        self.transformer = nn.ModuleList([
                            Block(dim=dim, num_heads=num_heads, 
                                  drop=drop, is_cross_attention=False) 
                            for _ in range(num_layers)
                            ])

        # get subsets, subset2id, and num_subsets
        self.create_subset2id()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.modality_embeddings = nn.Embedding(self.num_modalities, dim)
        self.num_patches_list = num_patches_list # number of patches in each modality
        self.pos_embeddings = nn.Parameter(torch.zeros(1, (1+sum(self.num_patches_list)), dim))
        num_classes = args.num_classes
        projection_dim = args[args.dataset_name].projection_dim
        self.classifier = nn.Linear(dim, num_classes)
        self.projection = nn.Linear(dim, projection_dim)
        self.register_buffer('prototypes', torch.zeros(num_classes, projection_dim))
       

        if args.checkpoint is None:
            trunc_normal_(self.pos_embeddings, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
            print('Initialize MissTransformer from scratch')

        if 'transformer_checkpoint' in args[args.dataset_name]:
            transformer_checkpoint = args[args.dataset_name].transformer_checkpoint
            ckpt = torch.load(transformer_checkpoint, map_location='cpu')
            state_dict = ckpt['state_dict']
            self.load_state_dict(state_dict, strict=False)
            print('Load MissTransformer checkpoint from {}'.format(transformer_checkpoint))

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


    def forward_train(self, x: Dict, mask: torch.Tensor, return_subsets_ids:bool=False):
        assert self.num_modalities == len(x)
        out = []
        out_mask = []

        # encoding and modality embedding and mask creation
        for i in range(self.num_modalities):
            tmp = self.m_encoders[i](x[f'm{i}'])  # (B, C, H, W)
            tmp = rearrange(tmp, 'b c h w -> b (h w) c')  # (B, H*W, C)
            tmp = self.m_norms[i](tmp)

            mask_i = mask[:, i].unsqueeze(-1).expand(tmp.shape[0], tmp.shape[1])   # (B, H*W)
            out_mask.append(mask_i)

            modality_embed = self.modality_embeddings(torch.full_like(mask_i, i).long().to(tmp.device))  # (B, H*W, C)
            # replace mask_i=True positions with zero tensors
            tmp = tmp * (~(mask_i.unsqueeze(-1))).float()
            tmp = tmp + modality_embed
            out.append(tmp)


        out = torch.cat(out, dim=1)   # (B, P, C)
        out_mask = torch.cat(out_mask, dim=1)  # (B, P)

        # create cls tokens
        B = mask.shape[0]
        subsets_ids = []
        if return_subsets_ids:
            for i in range(B):
                subsets_ids.append(self.subset2id[tuple(mask[i].tolist())])
        # subsets_ids = torch.tensor(subsets_ids).unsqueeze(1).to(out.device)
        # cls_tokens = self.cls_token_subsets(subsets_ids)  # (B, 1, C)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        out = torch.cat([cls_tokens, out], dim=1)
        out = out + self.pos_embeddings

        # masking 1 means invisible, assign a large negative value to the masked/invisible positions
        cls_mask = torch.zeros(B, 1).bool().to(mask.device)
        mask = torch.cat([cls_mask, out_mask], dim=1)
        mask = mask[:, None, None, :]
        mask = mask*(-1e9)
        assert out.shape[1] == mask.shape[-1]

        for block in self.transformer:
            out = block(out, mask=mask)
        feat = out[:, 0, :]
        out = self.classifier(feat)
        feat = self.projection(feat)
        if self.distance_metric == 'cosine_similarity':
            feat = F.normalize(feat, dim=-1)
        elif self.distance_metric == 'squared_euclidean':
            pass
        if return_subsets_ids:
            return out, feat, subsets_ids
        else:
            return out, feat


    def forward(self, x: Dict, mask: torch.Tensor):
        # start_time = time.time()    
        B = mask.shape[0]
        out, _ = self.forward_train(x, mask)
        end_time = time.time()
        # # report inference latency using ms
        # inference_time = (end_time - start_time)*1000
        # print(f'Inference time for batch size {B}: {inference_time:.4f} ms, per sample: {inference_time/B:.4f} ms')
        return out


if __name__ == "__main__":
    args = {'PolyMNIST':{"transformer_dim": 128, "transformer_heads": 4, "transformer_layers": 2, "transformer_drop": 0.0, 
                         "num_patches_list": [4, 4, 4], "num_masks": 0, 'projection_dim':64}, 
                         'dataset_name': 'PolyMNIST',
                "num_modalities": 3, "checkpoint": None, 'num_classes': 10}
    args = OmegaConf.create(args)
    model = DynamicTransformer(args)
    x = {'m0': torch.randn(2, 3, 28, 28), 'm1': torch.randn(2, 3, 28, 28), 'm2': torch.randn(2, 3, 28, 28)}
    mask = torch.tensor([[True, True, False], [False, True, True]])
    output, feat = model.forward_train(x, mask)
    print(output.shape)
    print(feat.shape)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

        

        
