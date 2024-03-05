"""transformer based denoiser"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
import bitnet
from tld.transformer_blocks import DecoderBlock, MLPSepConv, SinusoidalEmbedding, PositionalEmbedding


def l2norm(t, dim=-1):
    return F.normalize(t, dim=dim)

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) module.

    Args:
        dim (int): The input dimension.
        affine (bool, optional): If True, apply an affine transformation to the normalized output.
            Default is True.

    Attributes:
        scale (float): The scaling factor for the normalized output.
        gamma (torch.Tensor or float): The learnable parameter for the affine transformation.

    """

    def __init__(self, dim, affine=True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.0

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale

class DenoiserTransBlock(nn.Module):
    def __init__(self, 
                 patch_size=2, 
                 img_size=64, 
                 embed_dim=512, 
                 dropout=0, 
                 n_layers=6,
                 mlp_multiplier=4, 
                 n_channels=16
            ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        seq_len = int((self.img_size/self.patch_size)*((self.img_size/self.patch_size)))
        patch_dim = self.n_channels*self.patch_size*self.patch_size

        self.patchify_and_embed = nn.Sequential(
                                       nn.Conv2d(self.n_channels, patch_dim, kernel_size=self.patch_size, stride=self.patch_size),
                                       PositionalEmbedding(patch_dim, self.img_size),
                                       Rearrange('bs d h w -> bs (h w) d'),
                                       nn.LayerNorm(patch_dim),
                                       nn.Linear(patch_dim, self.embed_dim),
                                       nn.LayerNorm(self.embed_dim)
                                    )
        
        #self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        #self.register_buffer('precomputed_pos_enc', torch.arange(0, seq_len).long())

        self.decoder_blocks = nn.ModuleList([DecoderBlock(
                                            embed_dim=self.embed_dim,
                                            cond_dim=embed_dim,
                                            mlp_multiplier=self.mlp_multiplier,
                                            is_causal=False,
                                            dropout_level=self.dropout,
                                            mlp_class=MLPSepConv)
                                              for _ in range(self.n_layers)])

        self.out_proj = nn.Sequential(
            RMSNorm(self.embed_dim),
            nn.Linear(self.embed_dim, patch_dim)
        )


    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        b, c, h, w = x.shape
        pad_size = (
            0,
            math.ceil(w/self.patch_size)*self.patch_size-w,
            0,
            math.ceil(h/self.patch_size)*self.patch_size-h
        )
        pad = nn.ZeroPad2d(pad_size)
        x = pad(x)
        B, C, H, W = x.shape
        x = self.patchify_and_embed(x)
        #pos_enc = self.precomputed_pos_enc[:x.size(1)].expand(x.size(0), -1)
        #x = x+self.pos_embed(pos_enc)

        for block in self.decoder_blocks:
            x = block(x, cond, (H//self.patch_size, W//self.patch_size))
        x = self.out_proj(x)
        x = Rearrange(
                'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                h=int(H//self.patch_size),
                p1=self.patch_size, 
                p2=self.patch_size
            )(x)
        x = x[:, :, :h, :w]
        return x

class Denoiser(nn.Module):
    def __init__(self,
                 in_dim=16,
                 image_size=64, 
                 noise_embed_dims=128, 
                 cond_dim=512, 
                 patch_size=2, 
                 embed_dim=1024, 
                 dropout=0, 
                 n_layers=6,
                 ):
        super().__init__()

        self.in_dim = in_dim
        self.image_size = image_size
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim

        self.fourier_feats = nn.Sequential(SinusoidalEmbedding(embedding_dims=noise_embed_dims),
                                           nn.Linear(noise_embed_dims, self.embed_dim),
                                           nn.GELU(),
                                           nn.Linear(self.embed_dim, self.embed_dim)
                                           )

        self.denoiser_trans_block = DenoiserTransBlock(
            patch_size=patch_size, 
            img_size=image_size, 
            embed_dim=embed_dim,
            dropout=dropout, 
            n_layers=n_layers, 
            n_channels=in_dim
        )
        self.cond_proj = nn.Sequential(
            bitnet.BitLinear(cond_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, noise_level, cond):


        noise_level = self.fourier_feats(noise_level)[:, None, :]

        cond = self.cond_proj(cond)
        cond = torch.cat([noise_level, cond], dim=1) #bs, seq_len+1, d
        cond = self.norm(cond)

        x = self.denoiser_trans_block(x, cond)

        return x

"""
from tld.denoiser import Denoiser
import torch
model = Denoiser(16, 64, 16, 512, 2, 1024, 0, 2)
data, ctx, t = torch.randn(4, 16, 16, 8), torch.randn(4, 8, 512), torch.Tensor([0.1, 0.2, 0.3, 0.4])
out = model(data, t, ctx)
"""