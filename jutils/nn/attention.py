import math
import torch
import torch.nn as nn
from inspect import isfunction


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class QKVAttention(nn.Module):  
    """  
    A module which performs QKV attention.
    """
    def __init__(self, efficient_attn: bool = True, dropout: float = 0.0):
        super().__init__()  
        self.dropout = dropout  
        self.efficient_attn = efficient_attn  
        if self.efficient_attn:  
            try:  
                _ = nn.functional.scaled_dot_product_attention  
            except AttributeError:  
                print("Please update PyTorch to 2.0 or higher to use efficient attention.")  
                self.efficient_attn = False
  
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q, k, v: (n, ..., l, c) tensors of Queries, Keys, Values. The ...
                can be any number of batch dimensions (e.g. heads).
        Returns:
            res: (n, ..., l, c) tensor after attention.
        """
        if self.efficient_attn:  
            res = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        else:  
            ch = q.shape[-1]  
            scale = 1. / math.sqrt(ch)
            dot = torch.einsum('...td, ...kd -> ...tk', q, k) * scale  
            weight = torch.softmax(dot, dim=-1)  
            if self.dropout > 0.0:  
                weight = torch.dropout(weight, p=self.dropout, train=self.training)  
            res = torch.einsum('...dt, ...tv -> ...dv', weight, v)  
        return res  


class Attention(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        context_dim = default(context_dim, dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        
        self.qkv_attention = QKVAttention(dropout=attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        # get query from x
        B, N, C = x.shape
        q = self.to_q(x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # get key, value from context (default to self-attention if context is None)
        context = default(context, x)
        kv = self.to_kv(context).reshape(B, context.shape[1], 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        x = self.qkv_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class _TestAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """ Adapted from timm.models.vision_transformer.Attention for testing """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == "__main__":
    inner_dim = 4
    x = torch.randn((2, 3, inner_dim))

    # use the same weights for testing
    q_w = torch.randn((inner_dim, inner_dim))
    kv_w = torch.randn((inner_dim, inner_dim * 2)).T
    proj_w = torch.randn((inner_dim, inner_dim))
    proj_b = torch.randn((inner_dim))

    # Test Attention
    _attn = _TestAttention(dim=inner_dim, num_heads=2)
    _attn.qkv.weight.data = torch.cat((q_w, kv_w), dim=0)
    _attn.proj.weight.data = proj_w
    _attn.proj.bias.data = proj_b
    
    res = _attn(x)
    print("_TestAttention:", res.shape)
    print(res)

    # Test Self-Attention
    attn = Attention(dim=inner_dim, num_heads=2)
    attn.to_q.weight.data = q_w
    attn.to_kv.weight.data = kv_w
    attn.proj.weight.data = proj_w
    attn.proj.bias.data = proj_b
    
    res = attn(x, context=x)
    print("Attention:", res.shape)
    print(res)

    # Test Cross-Attention
    attn = Attention(dim=inner_dim, context_dim=16, num_heads=2)
    c = torch.randn((2, 4, 16))
    res = attn(x, context=c)
    print("Cross-Attention:", res.shape)
