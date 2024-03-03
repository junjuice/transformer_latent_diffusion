import torch
import torch.nn as nn
import bitnet
import numpy as np
from einops import rearrange

class SinusoidalEmbedding(nn.Module):
    def __init__(self, emb_min_freq=1.0, emb_max_freq=1000.0, embedding_dims=32):
        super(SinusoidalEmbedding, self).__init__()

        frequencies = torch.exp(
            torch.linspace(np.log(emb_min_freq), np.log(emb_max_freq),
                embedding_dims // 2))

        self.register_buffer('angular_speeds', 2.0 * torch.pi * frequencies)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x[:, None]
        embeddings = torch.cat([torch.sin(x * self.angular_speeds.repeat(x.shape[0], 1)),
                                torch.cos(x * self.angular_speeds.repeat(x.shape[0], 1))], dim=-1)
        return embeddings

class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_size):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(dim, max_size, max_size))

    def forward(self, x):
        return x + self.embedding[..., :x.shape[~1], :x.shape[~0]].expand_as(x)

class MHAttention(nn.Module):
    def __init__(self, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):

        assert q.size(-1) == k.size(-1)
        assert k.size(-2) == v.size(-2)

        q, k, v = [rearrange(x, 'bs n (h d) -> bs h n d', h=self.n_heads) for x in [q,k,v]]

        out = nn.functional.scaled_dot_product_attention(q, k, v,
                                                          attn_mask=attn_mask,
                                                          is_causal=self.is_causal,
                                                          dropout_p=self.dropout_level if self.training else 0)
        
        out = rearrange(out, 'bs h n d -> bs n (h d)', h=self.n_heads)

        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        """
        self.qkv_linear = nn.Linear(embed_dim, 3*embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)
        """
        assert n_heads % 2 == 0
        self.is_causal = is_causal
        self.attn = bitnet.BitMGQA(
            embed_dim=embed_dim,
            query_heads=n_heads,
            kv_heads=n_heads//2,
            dropout=dropout_level,
        )

    def forward(self, x):
        """
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        return self.mha(q, k, v)
        """
        return self.attn.forward(query=x,
                                 key=x,
                                 value=x,
                                 is_causal=self.is_causal)[0]

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        """
        self.kv_linear = nn.Linear(embed_dim, 2*embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)
        """
        assert n_heads % 2 == 0
        self.is_causal = is_causal
        self.attn = bitnet.BitMGQA(
            embed_dim=embed_dim,
            query_heads=n_heads,
            kv_heads=n_heads//2,
            dropout=dropout_level,
        )

    def forward(self, x, y):
        """
        q = self.q_linear(x)
        k, v = self.kv_linear(y).chunk(2, dim=2)
        return self.mha(q,k,v)
        """
        return self.attn.forward(query=x,
                                 key=y,
                                 value=y,
                                 is_causal=self.is_causal)[0]

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        super().__init__()
        """
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout_level)
        )
        """
        self.mlp = nn.Sequential(
            bitnet.BitFeedForward(
                dim=embed_dim,
                ff_mult=mlp_multiplier,
            ),
            nn.Dropout(dropout_level)
        )

    def forward(self, x, *args, **kwargs):
        return self.mlp(x)

class MLPSepConv(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        """see: https://github.com/ofsoundof/LocalViT"""
        super().__init__()
        self.pre_fc = bitnet.BitLinear(embed_dim, mlp_multiplier*embed_dim)
        self.post_fc = nn.Sequential(
            nn.GELU(),
            bitnet.BitLinear(mlp_multiplier*embed_dim, embed_dim),
            nn.Dropout(dropout_level)
        )
        self.conv = nn.Conv2d(
            mlp_multiplier*embed_dim, 
            mlp_multiplier*embed_dim, 
            kernel_size=3,
            padding='same', 
            groups=mlp_multiplier*embed_dim
        ) #<- depthwise conv

    def forward(self, x, size):
        h, w = size
        x = self.pre_fc(x)
        x = rearrange(x, 'bs (h w) d -> bs d h w', h=h, w=w)
        x = self.conv(x)
        x = rearrange(x, 'bs d h w -> bs (h w) d')
        x = self.post_fc(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, cond_dim, is_causal, mlp_multiplier, dropout_level, mlp_class=MLP):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim, is_causal, dropout_level, n_heads=embed_dim//64)
        self.cross_attention = CrossAttention(embed_dim, is_causal=False, dropout_level=dropout_level, n_heads=embed_dim//64)
        self.mlp = mlp_class(embed_dim, mlp_multiplier, dropout_level)
        self.cond_projection = bitnet.BitLinear(cond_dim, embed_dim)

    def forward(self, x, y, size):
        x = self.self_attention(x) + x
        x = self.cross_attention(x, self.cond_projection(y)) + x
        x = self.mlp(x, size) + x
        return x