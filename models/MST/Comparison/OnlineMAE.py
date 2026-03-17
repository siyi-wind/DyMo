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
from models.MST.networks.ConvNetworksImgMNIST import EncoderImg
from models.MST.networks.ConvNetworksTextMNIST import EncoderText
from models.MST.networks.ConvNetworksImgSVHN import EncoderSVHN


class OnlineMAE(nn.Module):
    '''
    Input is a dictionary of modalities and a mask indication matrix.
    Encode all modalities through modality-specific CNN encoders.
    Use the mask to select existing modalities and pass them through a transformer.
    Still input masked tokens into the transformer, use attention mask to avoid attending to masked tokens.
    Special embeddings: cls token (1,dim), modality embeddings (num_modalities, dim), intra-modality positional embeddings (1+sum_num_patches, dim)
    '''
    def __init__(self, args):
        super(OnlineMAE, self).__init__()
        num_text_features = len(args.alphabet)
        self.m_encoders = nn.ModuleDict({
                'mnist': EncoderImg(None),
                'svhn': EncoderSVHN(None),
                'text': EncoderText(None, num_text_features),
                })
        dim = args[args.dataset_name].transformer_dim
        num_heads = args[args.dataset_name].transformer_heads
        num_layers = args[args.dataset_name].transformer_layers
        drop = args[args.dataset_name].transformer_drop
        num_patches_list = args[args.dataset_name].num_patches_list
        recon_encoder_num_layers = args[args.dataset_name].recon_encoder_layers
        recon_decoder_num_layers = args[args.dataset_name].recon_decoder_layers
        self.modality_names = args.modality_names
        self.num_mem_tokens = args[args.dataset_name].num_mem_tokens

        self.m_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(args.num_modalities)])
        self.recon_encoder = nn.ModuleList([
                            Block(dim=dim, num_heads=num_heads, 
                                  drop=drop, is_cross_attention=False) 
                            for _ in range(recon_encoder_num_layers)
                            ])
        self.recon_decoder = nn.ModuleList([
                            Block(dim=dim, num_heads=num_heads, 
                                  drop=drop, is_cross_attention=False) 
                            for _ in range(recon_decoder_num_layers)
                            ])
        self.fusion_transformer = nn.ModuleList([
                            Block(dim=dim, num_heads=num_heads, 
                                  drop=drop, is_cross_attention=False) 
                            for _ in range(num_layers)
                            ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.num_modalities = args.num_modalities
        self.num_patches_list = num_patches_list # number of patches in each modality
        self.pos_embeddings = nn.Parameter(torch.zeros(1, (1+sum(self.num_patches_list)), dim))

        self.mae_pos_embeddings = nn.Parameter(torch.zeros(1, self.num_mem_tokens+(sum(self.num_patches_list)), dim))
        self.mem_token = nn.Parameter(torch.zeros(1, self.num_mem_tokens, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))  # mask token for reconstruction
        self.encoder_norm = nn.LayerNorm(dim)
        self.decoder_norm = nn.LayerNorm(dim)
        self.decoder_pred = nn.Linear(dim, dim)

        self.classifier = nn.Linear(dim, args.num_classes)
        self.criterion = nn.CrossEntropyLoss()

        if args[args.dataset_name].encoder_checkpoint:
            print(f'Load encoder checkpoint from {args[args.dataset_name].encoder_checkpoint}')
            encoder_checkpoint = args[args.dataset_name].encoder_checkpoint
            state_dict = torch.load(encoder_checkpoint, map_location='cpu')['state_dict']
            for module, module_name in zip([self.m_encoders], ['m_encoders.']):
                self.load_weights(module, module_name, state_dict)

        if not args.checkpoint:
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.mem_token, std=.02)
            trunc_normal_(self.mask_token, std=.02)
            trunc_normal_(self.pos_embeddings, std=.02)
            trunc_normal_(self.mae_pos_embeddings, std=.02)
            trunc_normal_(self.mask_token, std=.02)
            self.apply(self._init_weights)
            print("Random initialization")
        else:
            self.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=False)

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
    
    def forward_recon_encoder(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + pos  # (B, M+num_mem_tokens, C)
        for block in self.recon_encoder:
            x = block(x)
        x = self.encoder_norm(x)
        return x

    def forward_recon_decoder(self, x: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor, ids_restore: torch.Tensor):
        '''
        Replace masked positions with special mask tokens
        x (B, M'+num_mem, C); mask (B, M, C), pos (B, M+num_mem, C), ids_restore (B, M) '''
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + self.num_mem_tokens - x.shape[1], 1)   # (B, M+num_mem-(M'+num_mem), C)
        x_ = torch.cat([x[:, self.num_mem_tokens:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :self.num_mem_tokens, :], x_], dim=1) # append mem tokens

        # add positional embeddings
        x = x + pos  # (B, M+num_mem, C)
        for block in self.recon_decoder:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x[:, self.num_mem_tokens: , :])  # (B, M, C)
        return x

    def random_masking(self, x, pos_embedding, original_mask=None, random_masking=True):
        """
        From MAE, randomly mask a subset of the input sequence.
        x: (N, M, C)
        pos: (N, M, C)
        original_mask: (N, M), if provided, will be used to select existing modalities 1 means removed
        """
        if not random_masking:
            assert original_mask is not None, "original_mask should be provided when random_masking is False"
            # each row has the same number of 0s, i.e., same number of modalities
            assert original_mask.sum(dim=1).unique().numel() == 1, "Each sample should have the same number of modalities"
            N, L, D = x.shape
            # use the original mask to select existing modalities
            ids_keep = []
            ids_restore = []
            for i in range(N):
                m = original_mask[i]  # shape [n]
                keep_idx = torch.where(m == 0)[0]  
                remove_idx = torch.where(m == 1)[0]  
                # concat keep and remove indices
                ids = torch.cat([keep_idx, remove_idx], dim=0)  # [n]
                # create restore ids：ids_restore[i][ids[j]] = j
                restore = torch.argsort(ids)  # [n]
                ids_keep.append(keep_idx)
                ids_restore.append(restore)

            ids_keep = torch.stack(ids_keep)  # [N, M']
            ids_restore = torch.stack(ids_restore)  # [N, M]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            pos_embedding_masked = torch.gather(pos_embedding, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            # check if ids_restore is correct
            len_keep = ids_keep.shape[1]  
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)  # unshuffle to get the binary mask
            assert (mask == original_mask).all(), "The mask should be the same as the original mask"
            return x_masked, pos_embedding_masked, mask, ids_restore

        else:
            N, L, D = x.shape  # batch, length, dim
            # TODO Follow OnlineMAE to sample len_keep from [1, ..., L-1]
            len_keep = torch.randint(1, L, (1,), device=x.device).item()  # random sample from [1, ... , L-1]
            
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            pos_embedding_masked = torch.gather(pos_embedding, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
            return x_masked, pos_embedding_masked, mask, ids_restore
    

    def forward_fusion(self, x_origin: torch.Tensor, x_recon: torch.Tensor, mask: torch.Tensor):
        '''
        (B, P, C)  (B, P, C)  (B, P)  1 means invisible
        Replace visible positions with original tokens, append cls token, add positional embeddings
        '''
        # replace visible tokens with original tokens
        mask = mask.unsqueeze(-1)   # (B, P, 1)
        x = x_recon * mask + x_origin * (~mask)
        # Append cls token
        B, P, C = x_recon.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embeddings

        for block in self.fusion_transformer:
            x = block(x)
        return x
    
    def forward_loss(self, target: torch.Tensor, output: torch.Tensor, mask: torch.Tensor):
        '''
        (B, P, C)  (B, P, C)  (B, P) calculate the loss on mask=1 positions'''
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

        loss = (output - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def forward_recon(self, x: Dict, mask: torch.Tensor, random_masking: bool = True):
        assert self.num_modalities == len(x)
        mask = mask.bool()
        B, M = mask.shape

        # encoding and modality embedding and mask creation
        out = []
        for i, name in enumerate(self.modality_names):
            tmp = self.m_encoders[name](x[name])  # (B, C)
            tmp = tmp.unsqueeze(1)  # (B, 1, C)
            tmp = self.m_norms[i](tmp)
            out.append(tmp)
        
        out = torch.cat(out, dim=1)   # (B, M, C)
        recon_target = out.clone()  # (B, M, C)
        

        #### reconstruction 
        # TODO IMPORTANT: for reconstruction mask, 1 means removed
        mae_pos_embeddings = self.mae_pos_embeddings.expand(B, -1, -1)  # (B, M+num_mem_tokens, C)
        mem_pos, pos = mae_pos_embeddings[:, :self.num_mem_tokens, :], mae_pos_embeddings[:, self.num_mem_tokens:, :]  # (B, num_mem_tokens, C), (B, M, C)
        out, pos, mask, ids_restore = self.random_masking(out, pos, mask, random_masking)  # (B, M', C), (B, M', C), (B, M'), (B, M')
        # append mem tokens for features and pos_embeddings
        pos = torch.cat([mem_pos, pos], dim=1)  # [N, M'+num_mem, D]
        mem_token = self.mem_token
        mem_tokens = mem_token.expand(B, self.num_mem_tokens, -1)  # (B, num_mem_tokens, C)
        out = torch.cat([mem_tokens, out], dim=1)  # (B, M'+num_mem_tokens, C)

        # reconstruction encoder
        out = self.forward_recon_encoder(out, pos)
        out = self.forward_recon_decoder(out, mask, mae_pos_embeddings, ids_restore)  
        recon_loss = self.forward_loss(recon_target, out, mask)  

        # fusion
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, out], dim=1)
        x = x + self.pos_embeddings  # add positional embeddings
        for block in self.fusion_transformer:
            x = block(x)
        x = x[:, 0, :]  # take cls token
        x = self.classifier(x)  # (B, num_classes)
        return (recon_loss, x)
    
    def forward_train(self, x: Dict, mask: torch.Tensor, y: torch.Tensor = None):
        mask = mask.bool()
        assert (mask == False).all(), "Only support reconstruction for complete modalities, i.e., mask should be all False"
        recon_loss, out = self.forward_recon(x, mask, random_masking=True)
        # print(torch.softmax(out.detach(), dim=-1))
        ce_loss = self.criterion(out, y)
        loss = recon_loss + ce_loss
        return (loss, out)
    
    def forward(self, x: Dict, mask: torch.Tensor):
        _, x = self.forward_recon(x, mask, random_masking=False)
        return x


if __name__ == "__main__":
    alphabet_path = join(project_path, 'datasets/alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = {'MST':{"transformer_dim": 32, "transformer_heads": 2, "transformer_layers": 2, "transformer_drop": 0.0, "num_patches_list": [1, 1, 1], 
                         "recon_encoder_layers":1, "recon_decoder_layers": 1, "num_mem_tokens": 8,
                         'encoder_checkpoint':'/home/siyi/project/mm/result/Dynamic_project/MS12/whole_none_MST_MultiAE_0530_170327/downstream/checkpoint_best_acc.ckpt'}, 
                         "checkpoint": None,
                 'dataset_name': 'MST', 'alphabet': alphabet, 'modality_names': ['mnist', 'svhn', 'text'],
                "num_modalities": 3, "num_classes": 10}
    args = OmegaConf.create(args)
    model = OnlineMAE(args)
    x = {'mnist': torch.randn(2, 1, 28, 28), 'svhn': torch.randn(2, 3, 32, 32), 'text': torch.randn(2, 8, 71)}
    mask = torch.tensor([[False, False, False], [False, False, False]])
    loss, output = model.forward_train(x, mask, torch.tensor([1, 2]))
    for item in output:
        print(item.shape)
    print("Loss:", loss.item())
    mask = torch.tensor([[False, True, False], [True, False, False]])
    output = model.forward(x, mask)
    print(output.shape)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

        

        
