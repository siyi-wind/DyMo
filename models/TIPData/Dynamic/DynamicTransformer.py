from typing import Dict
import torch.nn as nn
import torch
from omegaconf import OmegaConf
import sys
from einops import rearrange,repeat
import json
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.utils.transformer import Block
from itertools import chain, combinations
import torch.nn.functional as F
from models.utils.TIP_utils.Transformer import TabularTransformerEncoder
from models.utils.TIP_utils.build_ssl_encoder import torchvision_ssl_encoder



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
        self.modality_names = args.modality_names
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

        dim = args[args.dataset_name].transformer_dim
        num_heads = args[args.dataset_name].transformer_heads
        num_layers = args[args.dataset_name].transformer_layers
        drop = args[args.dataset_name].transformer_drop
        num_patches_list = args[args.dataset_name].num_patches_list
        self.num_masks = args[args.dataset_name].num_masks
        self.num_modalities = args.num_modalities
        print('Randomly drop {} modalities'.format(self.num_masks))
        assert self.num_modalities == len(self.modality_names)

        self.m_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.num_modalities)])
        self.m_projection = nn.ModuleDict({
                'img': nn.Linear(2048, dim),
                'tabular': nn.Linear(512, dim)
                }) 
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
        if 'distance_metric' in args[args.dataset_name]:
            self.distance_metric = args[args.dataset_name].distance_metric
        else:
            self.distance_metric = 'cosine_similarity'
        print(f'Use distance metric for DynamicTransformer: {self.distance_metric}')

        # Load image and tabular encoders checkpoints

        if not args.checkpoint:
            trunc_normal_(self.pos_embeddings, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
            print('Initialize DynamicTransformer from scratch')

        if 'encoder_checkpoint' in args[args.dataset_name] and args[args.dataset_name].encoder_checkpoint:
            print(f'Load encoder checkpoint from {args[args.dataset_name].encoder_checkpoint}')
            encoder_checkpoint = args[args.dataset_name].encoder_checkpoint
            state_dict = torch.load(encoder_checkpoint, map_location='cpu')['state_dict']
            for module, module_name in zip(self.m_encoders.values(), ['encoder_imaging.', 'encoder_tabular.']):
                self.load_weights(module, module_name, state_dict)

        if 'transformer_checkpoint' in args[args.dataset_name]:
            transformer_checkpoint = args[args.dataset_name].transformer_checkpoint
            ckpt = torch.load(transformer_checkpoint, map_location='cpu')
            state_dict = ckpt['state_dict']
            self.load_state_dict(state_dict, strict=False)
            print('Load DynamicTransformer checkpoint from {}'.format(transformer_checkpoint))

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=.02)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()

    def load_weights(self, module, module_name, state_dict):
        state_dict_module = {}
        for k in list(state_dict.keys()):
            if k.startswith(module_name):
                state_dict_module[k[len(module_name):]] = state_dict[k]
        print(f'Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}')
        log = module.load_state_dict(state_dict_module, strict=True)
        assert len(log.missing_keys) == 0
    
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


    def forward_train(self, x: Dict, mask: torch.Tensor, return_subsets_ids:bool=False, return_origin_feat: bool=False):
        # assert self.num_modalities == len(x)
        out = []
        out_mask = []
        mask = mask.bool()  
        origin_mask = mask.clone().detach()  # (B, M)
        B, L = x['tabular'].shape
        with torch.no_grad():
            if 'tabular_missing_mask' in x:
                origin_mask_expanded = mask[:,1].unsqueeze(-1).expand(B, L)
                tabular_missing_mask = x['tabular_missing_mask'].bool()
                # only mask==True use tabular_missing_mask
                tabular_missing_mask = origin_mask_expanded & tabular_missing_mask
            else:
                tabular_missing_mask = mask[:,1].unsqueeze(-1).expand(B, L).clone() 

        # encoding and modality embedding and mask creation
        for i, name in enumerate(self.modality_names):
            if name == 'img':
                tmp = self.m_encoders[name](x[name])  
                tmp = rearrange(tmp[-1], 'b c h w -> b (h w) c')
                mask_i = mask[:, i].unsqueeze(-1).expand(tmp.shape[0], tmp.shape[1])   # (B, H*W)
            elif name == 'tabular':
                mask_i = tabular_missing_mask  # (B, P)
                tmp = x[name] * (~tabular_missing_mask).float()
                tmp = self.m_encoders[name](tmp, mask=tabular_missing_mask, mask_special=tabular_missing_mask)
                tmp = tmp[:, 1:]
            tmp = self.m_projection[name](tmp)  # (B, P, C)
            tmp = self.m_norms[i](tmp)

            # mask_i = mask[:, i].unsqueeze(-1).expand(tmp.shape[0], tmp.shape[1])   # (B, H*W)
            out_mask.append(mask_i)

            modality_embed = self.modality_embeddings(torch.full_like(mask_i, i).long().to(tmp.device))  # (B, H*W, C)
            # # TODO IMPORTANT replace mask_i=True positions with zero tensors
            tmp = tmp * (~(mask_i.unsqueeze(-1))).float()
            tmp = tmp + modality_embed
            out.append(tmp)


        out = torch.cat(out, dim=1)   # (B, P, C)
        out_mask = torch.cat(out_mask, dim=1)  # (B, P)

        # create cls tokens
        B = mask.shape[0]
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
        feat_origin = feat.clone().detach()
        out = self.classifier(feat)
        feat = self.projection(feat)
        if self.distance_metric == 'cosine_similarity':
            feat = F.normalize(feat, dim=-1)
        elif self.distance_metric == 'squared_euclidean':
            pass
        else:
            raise ValueError(f'Unknown distance metric: {self.distance_metric}')
        
        if return_origin_feat:
            return_feat = feat_origin
        else:
            return_feat = feat

        if return_subsets_ids:
            subsets_ids = []
            for i in range(B):
                subsets_ids.append(self.subset2id[tuple(origin_mask[i].tolist())])
            subsets_ids = torch.tensor(subsets_ids).unsqueeze(1).to(out.device)
            return out, return_feat, subsets_ids
        else:
            return out, return_feat

    def forward(self, x: Dict, mask: torch.Tensor):
        out, _ = self.forward_train(x, mask)
        return out


if __name__ == "__main__":
    alphabet_path = join(project_path, 'datasets/alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = {'DVM':{"transformer_dim": 256, "transformer_heads": 8, "transformer_layers": 2, "transformer_drop": 0.0, 
                         "num_patches_list": [16, 17], "num_masks": 0, 'projection_dim':128, 'distance_metric':'squared_euclidean',
                         'encoder_checkpoint': '/bigdata/siyi/data/result/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt',
                         }, 'DATA_field_lengths_tabular': '/bigdata/siyi/data/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
                         'dataset_name': 'DVM', 
                'alphabet': alphabet, 'modality_names': ['img', 'tabular'],
               "num_modalities": 2, "checkpoint": None, 'num_classes': 283, }

    args = OmegaConf.create(args)
    model = DynamicTransformer(args)
    x = {'img': torch.randn(2, 3, 128, 128), 'tabular': torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                    [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)}
    mask = torch.tensor([[False, True], [False, False]])
    output, feat, subsets_ids = model.forward_train(x, mask, return_subsets_ids=True, return_origin_feat=True)
    print(output.shape)
    print(feat.shape)
    print(subsets_ids)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

        
    # # test missing tabular
    # args.missing_tabular = True
    # x = {'img': torch.randn(2, 3, 128, 128), 'tabular': torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
    #                 [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32),
    #         'tabular_missing_mask': torch.tensor([[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False],
    #                                               [True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False]])}
    # mask = torch.tensor([[False, False], [False, True]])
    # output = model.forward(x, mask)
        
