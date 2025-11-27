import math
import torch
import torch.nn as nn
from pydoc import locate
from jaxtyping import Float
from functools import reduce
from einops import rearrange
from functools import partial
import torch.nn.functional as F


__all__ = [
    "zero_init", "LinearSwiGLU",
    "rms_norm", "RMSNorm", "AdaRMSNorm",
    "TokenMerge2D", "TokenSplitLast2D", "TokenMerge3D", "TokenSplitLast3D",
    "FourierFeatures", "TimestepEmbedder",
    "MappingFeedForwardBlock", "MappingNetwork", "FeedForwardBlock",
    "scale_for_cosine_sim", "AttentionBlock", "DimensionAttentionBlock", "CrossAttentionBlock", "TransformerLayer",
    "RegisterAttentionBlock", "RegisterCrossAttentionBlock", "TransformerLayerWithRegisters",
]
# ===================================================================================================


COMPILE = False
if torch.cuda.is_available():
    compile_fn = partial(torch.compile, fullgraph=False, backend='inductor' if torch.cuda.get_device_capability()[0] >= 7 else 'aot_eager')
else:
    compile_fn = lambda f: f


# ===================================================================================================


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        x = x @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(cond_features, features, bias=False))

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        proj_out = self.linear(cond)
        if proj_out.ndim == 2:          # (bs, dim) -> (bs, 1, dim)
            proj_out = proj_out[:, None, :]
        else:
            assert proj_out.shape[1] == x.shape[1] or proj_out.shape[1] == 1, \
                f"mismatch in AdaRMSNorm shape: x={x.shape} proj_out={cond.shape}"
        return rms_norm(x, proj_out + 1, self.eps)


# ===================================================================================================
# patchification


class TokenMerge2D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        if isinstance(patch_size, int): patch_size = (patch_size, patch_size)
        self.ph = patch_size[0]
        self.pw = patch_size[1]
        self.proj = nn.Linear(in_features * self.ph * self.pw, out_features, bias=False)

    def forward(self, x, pos):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.ph, nw=self.pw)
        pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=self.ph, nw=self.pw)
        return self.proj(x), torch.mean(pos, dim=-2)


class TokenSplitLast2D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2), zero_init: bool = True):
        super().__init__()
        if isinstance(patch_size, int): patch_size = (patch_size, patch_size)
        self.ph = patch_size[0]
        self.pw = patch_size[1]
        self.norm = RMSNorm(in_features)
        self.proj = nn.Linear(in_features, out_features * self.ph * self.pw, bias=False)
        if zero_init: nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = rearrange(x, "... h w (ph pw c) -> ... (h ph) (w pw) c", ph=self.ph, pw=self.pw)
        return x


class TokenMerge3D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features * self.t * self.h * self.w, out_features, bias=False)

    def forward(self, x, pos, **kwargs):
        x = rearrange(x, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw e)", nt=self.t, nh=self.h, nw=self.w)
        pos = rearrange(pos, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw) e", nt=self.t, nh=self.h, nw=self.w)

        return self.proj(x), torch.mean(pos, dim=-2)


class TokenSplitLast3D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features, out_features * self.t * self.h * self.w, bias=False)
        self.norm = RMSNorm(in_features)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.proj(x)
        x = rearrange(x, "... t h w (nt nh nw e) -> ... (t nt) (h nh) (w nw) e", nt=self.t, nh=self.h, nw=self.w)
        return x


# ===================================================================================================
# fourier/time embedding


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, depth: int, dim_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = dim
        self.depth = depth
        self.dim_mlp = dim_mlp
        self.dropout = dropout

        self.time_emb = FourierFeatures(1, dim)
        self.time_in_proj = nn.Linear(dim, dim, bias=False)
        self.mapping = MappingNetwork(depth, dim, dim_mlp, dropout=dropout)

        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(self, t: Float[torch.Tensor, 'b']) -> Float[torch.Tensor, 'b dim']:
        if t.ndim == 1:
            t = t[..., None]
        time_emb = self.time_in_proj(self.time_emb(t))
        time_emb = self.mapping(time_emb)
        return time_emb


# ===================================================================================================


class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = LinearSwiGLU(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, d_cond_norm=None, dropout=0.0):
        super().__init__()
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.up_proj = LinearSwiGLU(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x, cond_norm=None):
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


# ===================================================================================================
# Attention


def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int = 64,
        d_cond_norm: int | None = None,
        dropout: float = 0.0,
        rope_cls: str = 'jutils.nn.rope.AxialRoPE2D',
    ):
        super().__init__()
        self.d_head = d_head
        self.d_model = d_model
        self.n_heads = d_model // d_head
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))

        self.pos_emb = None
        if rope_cls is not None:
            self.pos_emb = locate(rope_cls)(d_head, self.n_heads, relative_canvas=True, learnable_freqs=False)

    def forward(self, x, pos, cond_norm=None):
        skip = x

        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)

        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)

        if self.pos_emb is not None:
            pos = pos.to(qkv.dtype)
            theta = self.pos_emb(pos)
            theta = theta.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
            k = self.pos_emb.apply_emb(k, theta)

        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip
    

class DimensionAttentionBlock(AttentionBlock):
    """
    Expects (b, ..., dim), reshapes in forward, and applies attention over all '...' dimensions.
    This allows you to apply AdaRMSNorm only to specific dimensions.
    """
    def forward(
        self,
        x: Float[torch.Tensor, 'b ... c'],
        pos: Float[torch.Tensor, 'b ... d'],
        cond_norm: Float[torch.Tensor, 'b ... e'] = None,
    ):
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)

        B, *DIMS, C = x.shape
        qkv = self.qkv_proj(x)
        x = rearrange(x, "b ... c -> b (...) c")
        pos = rearrange(pos, "b ... c -> b (...) c")
        qkv = rearrange(qkv, "b ... c -> b (...) c")
        pos = pos.to(qkv.dtype)
        theta = self.pos_emb(pos)

        q, k, v = rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
        theta = theta.movedim(-2, -3)
        q = self.pos_emb.apply_emb(q, theta)
        k = self.pos_emb.apply_emb(k, theta)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        x = x.view(B, *DIMS, C)
        return x + skip


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cross: int,
        d_head: int = 64,
        d_cond_norm: int | None = None,
        dropout: float = 0.0,
        rope_cls: str = 'jutils.nn.rope.AxialRoPE2D',
    ):
        super().__init__()
        self.d_head = d_head
        self.d_model = d_model
        self.n_heads = d_model // d_head
        # TODO check if ada norm makes sense for keys and values
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.norm_cross = RMSNorm(d_cross)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_cross, d_model * 2, bias=False)
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))

        self.pos_emb = None
        if rope_cls is not None:
            self.pos_emb = locate(rope_cls)(d_head, self.n_heads, relative_canvas=True, learnable_freqs=False)

    def forward(
        self,
        x: Float[torch.Tensor, "b l d"],
        pos: Float[torch.Tensor, "b l nc"],
        x_cross: Float[torch.Tensor, "b l' d'"],
        cond_norm: Float[torch.Tensor, "b d"] | None = None,
    ) -> Float[torch.Tensor, "b ... d"]:
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x_cross = self.norm_cross(x_cross)
        q = self.q_proj(x)
        kv = self.kv_proj(x_cross)

        q = rearrange(q, "n l (nh e) -> n nh l e", e=self.d_head)
        k, v = rearrange(kv, "n l (t nh e) -> t n nh l e", t=2, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)

        if self.pos_emb is not None:
            pos = pos.to(q.dtype)
            theta = self.pos_emb(pos)
            theta = theta.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
        
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip
    

class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_cross=None,
        d_head=64,
        d_cond_norm=None,
        dropout=0.0,
        ff_expand=3,
        rope_cls='jutils.nn.rope.AxialRoPE2D',
        compile: bool = False
    ):
        super().__init__()
        global COMPILE
        COMPILE = compile
        
        d_ff = d_model * ff_expand

        self.self_attn = AttentionBlock(
            d_model=d_model,
            d_head=d_head,
            d_cond_norm=d_cond_norm,
            dropout=dropout,
            rope_cls=rope_cls,
        )

        self.cross_attn = None
        if d_cross is not None:
            self.cross_attn = CrossAttentionBlock(
                d_model=d_model,
                d_cross=d_cross,
                d_head=d_head,
                d_cond_norm=d_cond_norm,
                dropout=dropout,
                rope_cls=rope_cls,
            )
        
        self.ff = FeedForwardBlock(d_model, d_ff, d_cond_norm, dropout)

        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(
        self,
        x: Float[torch.Tensor, "b n c"],
        pos: Float[torch.Tensor, "b n d"],
        cond_norm: Float[torch.Tensor, "b 1|n e"] = None,
        x_cross: Float[torch.Tensor, "b m k"] = None,
    ):
        x = self.self_attn(x, pos, cond_norm=cond_norm)
        if self.cross_attn is not None:
            x = self.cross_attn(x, pos, x_cross=x_cross, cond_norm=cond_norm)
        x = self.ff(x, cond_norm=cond_norm)
        return x


# ===================================================================================================


class RegisterAttentionBlock(AttentionBlock):
    """ [register tokens, ... other tokens] """
    def __init__(self, *args, n_registers: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_registers = n_registers
        assert self.n_registers >= 0, "n_registers must be non-negative"

    def forward(self, x, pos, cond_norm=None):
        skip = x

        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)

        qkv = self.qkv_proj(x)
        pos = pos.to(qkv.dtype)
        theta = self.pos_emb(pos)

        q, k, v = rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)

        # exclude register tokens from RoPE (registers first)
        if self.n_registers > 0:
            q_r = q[:, :, :self.n_registers, :]
            k_r = k[:, :, :self.n_registers, :]
            q = q[:, :, self.n_registers:, :]
            k = k[:, :, self.n_registers:, :]

            theta = theta.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
            k = self.pos_emb.apply_emb(k, theta)

            # concatenate back
            q = torch.cat([q_r, q], dim=-2)
            k = torch.cat([k_r, k], dim=-2)
        else:
            theta = theta.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
            k = self.pos_emb.apply_emb(k, theta)

        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class RegisterCrossAttentionBlock(CrossAttentionBlock):
    """ [register tokens, ... other tokens] """
    def __init__(self, *args, n_registers: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_registers = n_registers
        assert self.n_registers >= 0, "n_registers must be non-negative"

    def forward(
        self,
        x: Float[torch.Tensor, "b l d"],
        pos: Float[torch.Tensor, "b l nc"],
        x_cross: Float[torch.Tensor, "b l' d'"],
        cond_norm: Float[torch.Tensor, "b d"] | None = None,
    ) -> Float[torch.Tensor, "b ... d"]:
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)

        x_cross = self.norm_cross(x_cross)
        kv = self.kv_proj(x_cross)
        q = self.q_proj(x)

        pos = pos.to(q.dtype)
        theta = self.pos_emb(pos)

        q = rearrange(q, "n l (nh e) -> n nh l e", e=self.d_head)
        k, v = rearrange(kv, "n l (t nh e) -> t n nh l e", t=2, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)

        theta = theta.movedim(-2, -3)
        if self.n_registers > 0:
            q_r = q[:, :, :self.n_registers, :]
            q = q[:, :, self.n_registers:, :]
            q = self.pos_emb.apply_emb(q, theta)
            q = torch.cat([q_r, q], dim=-2)
        else:
            q = self.pos_emb.apply_emb(q, theta)
        
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class TransformerLayerWithRegisters(nn.Module):
    def __init__(
        self,
        d_model,
        d_cross=None,
        d_head=64,
        d_cond_norm=None,
        dropout=0.0,
        ff_expand=3,
        rope_cls='jutils.nn.rope.AxialRoPE2D',
        compile: bool = False,
        n_registers: int = 1,
    ):
        super().__init__()
        global COMPILE
        COMPILE = compile
        self.n_registers = n_registers
        
        d_ff = d_model * ff_expand

        self.self_attn = RegisterAttentionBlock(
            d_model=d_model,
            d_head=d_head,
            d_cond_norm=d_cond_norm,
            dropout=dropout,
            rope_cls=rope_cls,
            n_registers=n_registers,
        )

        self.cross_attn = None
        if d_cross is not None:
            self.cross_attn = RegisterCrossAttentionBlock(
                d_model=d_model,
                d_cross=d_cross,
                d_head=d_head,
                d_cond_norm=d_cond_norm,
                dropout=dropout,
                rope_cls=rope_cls,
                n_registers=n_registers,
            )
        
        self.ff = FeedForwardBlock(d_model, d_ff, d_cond_norm, dropout)

        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(
        self,
        x: Float[torch.Tensor, "b n c"],
        pos: Float[torch.Tensor, "b n d"],
        cond_norm: Float[torch.Tensor, "b 1|n e"] = None,
        x_cross: Float[torch.Tensor, "b m k"] = None,
    ):
        x = self.self_attn(x, pos, cond_norm=cond_norm)
        if self.cross_attn is not None:
            x = self.cross_attn(x, pos, x_cross=x_cross, cond_norm=cond_norm)
        x = self.ff(x, cond_norm=cond_norm)
        return x


if __name__ == "__main__":
    transformer = TransformerLayer(768, d_cond_norm=128, d_cross=64)
    kwargs = dict(
        x=torch.randn((1, 256, 768)),
        pos=torch.randn((1, 256, 2)),
        cond_norm=torch.randn((1, 1, 128)),
        x_cross=torch.randn((1, 256, 64)),
    )
    out = transformer(**kwargs)
    print(out.shape)
