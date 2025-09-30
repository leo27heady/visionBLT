import torch
import numpy as np
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

RMSNorm = nn.RMSNorm


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class ReAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    
class LeFF(nn.Module):
    
    def __init__(self, dim = 192, scale = 4, depth_kernel = 3):
        super().__init__()
        
        scale_dim = dim*scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(scale_dim),
                                    nn.GELU(),
                                    Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                    )
        
        self.depth_conv =  nn.Sequential(nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
                          nn.BatchNorm2d(scale_dim),
                          nn.GELU(),
                          Rearrange('b c h w -> b (h w) c', h=14, w=14)
                          )
        
        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(dim),
                                    nn.GELU(),
                                    Rearrange('b c n -> b n c')
                                    )
        
    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x
    
    
class LCAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = q[:, :, -1, :].unsqueeze(2) # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class PredictionHead(nn.Module):
    def __init__(self, dim, pixel_vocab_size=256, norm_eps=1e-5):
        super().__init__()

        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(
            dim,
            pixel_vocab_size,
            bias=False,
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_preds = self.norm(h)
        h_preds = self.output(h_preds)
        h_preds = h_preds.float()
        return h_preds
  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, dim = 192, depth = 4, heads = 3, in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()

        self.dim = dim

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        h_w = image_size // patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size, h=h_w, w=h_w, c=in_channels),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)

        self.to_output_embedding = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b t (h w) (p1 p2 c) -> b t c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=h_w, w=h_w, c=in_channels),
        )

        # self.head = nn.Sequential(
        #     Rearrange('b t (h w) (p1 p2 c) -> b (t h w p1 p2) c', p1=patch_size, p2=patch_size, h=h_w, c=16),
        #     PredictionHead(dim=16),
        # )

        self.apply(self._init_weights)

    def forward(self, x, patch_lengths=None, ngram_ids=None):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        x += self.pos_embedding[:, :, :n]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x, '(b t) ... -> b t ...', b=b)
        
        x = rearrange(x, 'b t n d -> b (t n) d')
        x = self.temporal_transformer(x)
        x = rearrange(x, 'b (t n) d -> b t n d', t=t)

        # x = rearrange(x, 'b t n d -> b (t n) d', t=t)
        x = self.to_output_embedding(x)
        # x = self.head(x)

        return x

    def _init_weights(self, module):
        init_std = self.dim ** (-0.5)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def init_weights(self):
        pass
