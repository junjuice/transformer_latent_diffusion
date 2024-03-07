"""transformer based denoiser"""
from copy import deepcopy
from typing import Optional
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

class ModelEmaV3(nn.Module):
    """ Model Exponential Moving Average V3

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V3 of this module leverages for_each and in-place operations for faster performance.

    Decay warmup based on code by @crowsonkb, her comments:
      If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
      good values for models you plan to train for a million or more steps (reaches decay
      factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
      you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
      215.4k steps).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(
            self,
            model,
            decay: float = 0.9999,
            min_decay: float = 0.0,
            update_after_step: int = 0,
            use_warmup: bool = False,
            warmup_gamma: float = 1.0,
            warmup_power: float = 2/3,
            device: Optional[torch.device] = None,
            foreach: bool = True,
            exclude_buffers: bool = False,
    ):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.foreach = foreach
        self.device = device  # perform ema on different device from model if set
        self.exclude_buffers = exclude_buffers
        if self.device is not None and device != next(model.parameters()).device:
            self.foreach = False  # cannot use foreach methods with different devices
            self.module.to(device=device)

    def get_decay(self, step: Optional[int] = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if step is None:
            return self.decay

        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0

        if self.use_warmup:
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            decay = max(min(decay, self.decay), self.min_decay)
        else:
            decay = self.decay

        return decay

    @torch.no_grad()
    def update(self, model, step: Optional[int] = None):
        decay = self.get_decay(step)
        if self.exclude_buffers:
            self.apply_update_no_buffers_(model, decay)
        else:
            self.apply_update_(model, decay)

    def apply_update_(self, model, decay: float):
        # interpolate parameters and buffers
        if self.foreach:
            ema_lerp_values = []
            model_lerp_values = []
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    ema_lerp_values.append(ema_v)
                    model_lerp_values.append(model_v)
                else:
                    ema_v.copy_(model_v)

            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(ema_lerp_values, model_lerp_values, weight=1. - decay)
            else:
                torch._foreach_mul_(ema_lerp_values, scalar=decay)
                torch._foreach_add_(ema_lerp_values, model_lerp_values, alpha=1. - decay)
        else:
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    ema_v.lerp_(model_v, weight=1. - decay)
                else:
                    ema_v.copy_(model_v)

    def apply_update_no_buffers_(self, model, decay: float):
        # interpolate parameters, copy buffers
        ema_params = tuple(self.module.parameters())
        model_params = tuple(model.parameters())
        if self.foreach:
            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(ema_params, model_params, weight=1. - decay)
            else:
                torch._foreach_mul_(ema_params, scalar=decay)
                torch._foreach_add_(ema_params, model_params, alpha=1 - decay)
        else:
            for ema_p, model_p in zip(ema_params, model_params):
                ema_p.lerp_(model_p, weight=1. - decay)

        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(model_b.to(device=self.device))

    @torch.no_grad()
    def set(self, model):
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(model_v.to(device=self.device))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)