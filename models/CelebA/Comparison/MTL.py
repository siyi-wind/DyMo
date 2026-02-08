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
from itertools import chain, combinations
from models.CelebA.networks.ConvNetworksImgCelebA import EncoderImg
from models.CelebA.networks.ConvNetworksTextCelebA import EncoderText





def powerset(iterable):
    "powerset([1,2,3]) --> (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class MTL(nn.Module):
    '''
    Multi-task Transformer. Paper: Are Multimodal Transformers Robust to Missing Modality?
    - Input is a dictionary of modalities and a mask indication matrix.
    - Encode all modalities through modality-specific CNN encoders.
    - Each modality subset has a different CLS token
    - Mask CLS token's attention to missing modalities.
    '''
    def __init__(self, args):
        super(MTL, self).__init__()
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
        self.num_masks = args[args.dataset_name].num_masks
        self.modality_names = args.modality_names
        print('Randomly drop {} modalities'.format(self.num_masks))

        # get subsets, subset2id, and num_subsets
        self.num_modalities = args.num_modalities
        self.create_subset2id()

        self.m_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(args.num_modalities)])
        self.transformer = nn.ModuleList([
                            Block(dim=dim, num_heads=num_heads, 
                                  drop=drop, is_cross_attention=False) 
                            for _ in range(num_layers)
                            ])
        self.cls_token_subsets = nn.Embedding(self.num_subsets, dim)
        self.modality_embeddings = nn.Embedding(args.num_modalities, dim)
        self.num_modalities = args.num_modalities
        self.num_patches_list = num_patches_list # number of patches in each modality
        self.pos_embeddings = nn.Parameter(torch.zeros(1, (1+sum(self.num_patches_list)), dim))
        self.classifier = nn.Linear(dim, args.num_classes)
        self.criterion = nn.CrossEntropyLoss()

        if not args.checkpoint:
            trunc_normal_(self.pos_embeddings, std=.02)
            self.apply(self._init_weights)
            print('Initialize MTL from scratch')
        else:
            self.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['state_dict'], strict=False)

        if 'transformer_checkpoint' in args[args.dataset_name]:
            transformer_checkpoint = args[args.dataset_name].transformer_checkpoint
            ckpt = torch.load(transformer_checkpoint, map_location='cpu')
            state_dict = ckpt['state_dict']
            self.load_state_dict(state_dict, strict=False)
            print('Load MTL checkpoint from {}'.format(transformer_checkpoint))

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


    def forward_one_train(self, x: Dict, mask: torch.Tensor):
        # print('Use random dropout mask')
        assert self.num_modalities == len(x)
        out = []
        out_mask = []
        if self.num_masks > 0:
            mask = mask.bool()
            assert (~mask).sum(dim=1).min() > 1
            noise = torch.rand(mask.shape, device=mask.device) * (~mask)
            ids_shuffle = torch.argsort(noise, dim=1, descending=True)
            ids_second_mask = ids_shuffle[:, :self.num_masks]
            mask = mask.scatter(1, ids_second_mask, True)

        # encoding and modality embedding and mask creation
        for i, name in enumerate(self.modality_names):
            tmp = self.m_encoders[name](x[name])  
            tmp = tmp.unsqueeze(1)  # (B, 1, C)
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
        for i in range(B):
            subsets_ids.append(self.subset2id[tuple(mask[i].tolist())])
        subsets_ids = torch.tensor(subsets_ids).unsqueeze(1).to(out.device)
        cls_tokens = self.cls_token_subsets(subsets_ids)  # (B, 1, C)
        out = torch.cat([cls_tokens, out], dim=1)
        out = out + self.pos_embeddings

        # masking cls tokens attention to missing modalities
        # masking 1 means invisible, assign a large negative value to the masked/invisible positions
        cls_mask = torch.zeros(B, 1).bool().to(mask.device)
        mask = torch.cat([cls_mask, out_mask], dim=1)
        attention_mask = torch.zeros(B, mask.shape[1], mask.shape[1], dtype=torch.bool, device=mask.device)
        attention_mask[:, 0, :] = mask   # mask only impact cls token
        mask = attention_mask[:,None, :, :]
        mask = mask*(-1e9)
        assert out.shape[1] == mask.shape[-1]

        for block in self.transformer:
            out = block(out, mask=mask)
        out = out[:, 0, :]
        out = self.classifier(out)
        return out

    def forward(self, x: Dict, mask: torch.Tensor):
        assert self.num_modalities == len(x)
        out = []
        out_mask = []

        # encoding and modality embedding and mask creation
        for i, name in enumerate(self.modality_names):
            tmp = self.m_encoders[name](x[name])  
            tmp = tmp.unsqueeze(1)  # (B, 1, C)
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
        for i in range(B):
            subsets_ids.append(self.subset2id[tuple(mask[i].tolist())])
        subsets_ids = torch.tensor(subsets_ids).unsqueeze(1).to(out.device)
        cls_tokens = self.cls_token_subsets(subsets_ids)  # (B, 1, C)
        out = torch.cat([cls_tokens, out], dim=1)
        out = out + self.pos_embeddings

        # masking cls tokens attention to missing modalities
        # masking 1 means invisible, assign a large negative value to the masked/invisible positions
        cls_mask = torch.zeros(B, 1).bool().to(mask.device)
        mask = torch.cat([cls_mask, out_mask], dim=1)
        attention_mask = torch.zeros(B, mask.shape[1], mask.shape[1], dtype=torch.bool, device=mask.device)
        attention_mask[:, 0, :] = mask   # mask only impact cls token
        mask = attention_mask[:,None, :, :]
        mask = mask*(-1e9)
        assert out.shape[1] == mask.shape[-1]

        for block in self.transformer:
            out = block(out, mask=mask)
        out = out[:, 0, :]
        out = self.classifier(out)
        return out

    def forward_train(self, x: Dict, mask: torch.Tensor, y: torch.Tensor):
        '''Calculate CE loss over all subsets'''
        loss_list = []
        for k, v in self.id2subset.items():
            mask = torch.tensor(v, dtype=torch.bool, device=y.device)
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
    args = {'CelebA':{"transformer_dim": 32, "transformer_heads": 2, "transformer_layers": 2, "transformer_drop": 0.0, 
                         "num_patches_list": [1, 1], "num_masks": 0}, 
                         'dataset_name': 'CelebA', 'alphabet': alphabet, 'modality_names': ['img', 'text'],
                "num_modalities": 2, "checkpoint": None, 'num_classes': 2}
    args = OmegaConf.create(args)
    model = MTL(args)
    x = {'img': torch.randn(2, 3, 64, 64), 'text': torch.randn(2, 256, 71)}
    mask = torch.tensor([[True, False], [True, False]])
    loss, out = model.forward_train(x, mask, torch.tensor([0, 1]))
    print("Loss:", loss.item())
    print("Output shape:", out.shape)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

        

        
