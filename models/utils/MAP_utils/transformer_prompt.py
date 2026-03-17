import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, prompts=None, learnt_p=False, prompt_type='input'):
        B, N, C = x.shape
        if prompts is None or prompt_type == 'input':
            # origin attention
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)
            start_pos = mask.size(1) - N
            mask = mask[:,start_pos:]

        elif prompt_type == 'attention':
            # prefix prompt tuning
            P = prompts.size(1)
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, C)
            )   
            
            if learnt_p:
                P = P//2
                prompts_k = prompts[:,:P]
                prompts_v = prompts[:,P:]
            else:
                prompts_k = prompts
                prompts_v = prompts

            q, k, v = (
                qkv[:,:,0,:].reshape(B,N,12,C//12).permute(0,2,1,3),
                torch.cat([prompts_k,qkv[:,:,1,:]], dim=1).reshape(B,N+P,12,C//12).permute(0,2,1,3),
                torch.cat([prompts_v,qkv[:,:,2,:]], dim=1).reshape(B,N+P,12,C//12).permute(0,2,1,3),
            )  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None, prompts=None, learnt_p=False, prompt_type='input'):
        if prompts is not None and prompt_type == 'input':
            x = torch.cat([prompts,x],dim=1)        
        _x, attn = self.attn(self.norm1(x), mask=mask, prompts=prompts, learnt_p=learnt_p, prompt_type=prompt_type)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn
