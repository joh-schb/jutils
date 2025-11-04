from functools import reduce

import torch
from torch import nn
import math
from abc import ABC, abstractmethod


__all__ = [
    "centers", "make_grid", "bounding_box", "AxialRoPEBase",
    "AxialRoPE1D", "make_axial_pos_1d",
    "AxialRoPE2D", "make_axial_pos_2d",
    "AxialRoPE3D", "make_axial_pos_3d",
]
# ===============================================================================================


def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)


def bounding_box(h, w, pixel_aspect_ratio=1.0):
    # Adjusted dimensions
    w_adj = w
    h_adj = h * pixel_aspect_ratio

    # Adjusted aspect ratio
    ar_adj = w_adj / h_adj

    # Determine bounding box based on the adjusted aspect ratio
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return y_min, y_max, x_min, x_max


class AbstractPosEnc(nn.Module, ABC):
    def __init__(self, d_head, n_heads):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads

    @abstractmethod
    def forward(self, pos):
        pass

    @abstractmethod
    def apply_emb(self, x, theta):
        pass


class AxialRoPEBase(AbstractPosEnc):
    def __init__(self, d_head, n_heads, in_place=False):
        super().__init__(d_head, n_heads)
        self.in_place = in_place

    def apply_emb(self, x, theta):
        if self.in_place:
            return apply_rotary_emb_(x, theta)
        else:
            return apply_rotary_emb(x, theta)

    @abstractmethod
    def forward(self, pos):
        pass


def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class ApplyRotaryEmbeddingInplacePrecomputed(torch.autograd.Function):
    @staticmethod
    def _apply_rotary_emb_inplace_precomputed(x, sin, cos, conj):
        dtype = reduce(torch.promote_types, (x.dtype, sin.dtype, torch.float32))
        d = sin.shape[-1]
        assert d * 2 <= x.shape[-1]
        x1, x2 = x[..., :d], x[..., d : d * 2]
        x1_, x2_, cos, sin = x1.to(dtype), x2.to(dtype), cos.to(dtype), sin.to(dtype)
        sin = -sin if conj else sin
        y1 = x1_ * cos - x2_ * sin
        y2 = x2_ * cos + x1_ * sin
        x1.copy_(y1)
        x2.copy_(y2)

    @staticmethod
    def forward(x, sin, cos, conj):
        ApplyRotaryEmbeddingInplacePrecomputed._apply_rotary_emb_inplace_precomputed(x, sin, cos, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, sin, cos, conj = inputs
        ctx.save_for_backward(sin, cos)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        sin, cos = ctx.saved_tensors
        ApplyRotaryEmbeddingInplacePrecomputed._apply_rotary_emb_inplace_precomputed(
            grad_output, sin, cos, conj=not ctx.conj
        )
        return grad_output, None, None, None


class AxialRoPE2D(AxialRoPEBase):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        learnable_freqs: bool = False,
        relative_canvas: bool = True,
        in_place: bool = False,
        half_embedding: bool = True,
    ):
        # super().__init__(dim, n_heads, in_place=not learnable_freqs)  # Sometimes problematic, disabled for now

        # we only apply RoPE to half of the token
        if half_embedding:
            assert dim % 2 == 0, "Half embedding is only supported for even dimensions"
            dim //= 2

        super().__init__(dim, n_heads, in_place=in_place)
        if learnable_freqs:
            assert not in_place, "In-place RoPE with learnable frequencies is not supported"
        self.learnable_freqs = learnable_freqs

        if relative_canvas:
            min_freq = math.pi
            max_freq = 10.0 * math.pi
        else:
            min_freq = 1 / 10_000
            max_freq = 1.0

        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        # 2 * 2 for sin and cos, height and width
        freqs = torch.stack([torch.linspace(log_min, log_max, n_heads * dim // (2 * 2) + 1)[:-1].exp()] * 2)
        self.freqs = nn.Parameter(freqs.view(2, dim // (2 * 2), n_heads).mT.contiguous(), requires_grad=learnable_freqs)

    def extra_repr(self):
        return f"dim={self.freqs.shape[-1] * (2 * 2)}, n_heads={self.freqs.shape[-2]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs[0].to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs[1].to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


# Cached version of AxialRoPE2D. DO NOT USE UNLESS YOU'RE SURE IT'S VALID IN YOUR CASE
# This asumes that the position tensor never changes (incl. its exact value) during training
class CachedAxialRoPE2D(AxialRoPE2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.learnable_freqs
        assert self.in_place
        self.register_buffer("cache", None)

    def forward(self, *args, **kwargs):
        # TODO: this type of on-the-fly caching is actually quite suboptimal
        # The problem is that assigning variables to modules causes graph breaks,
        # and even though we only do that for the first forward pass,
        # that's what's taken into consideration during compilation
        # In practice, this will cause ~100us of overhead per forward
        if self.cache is None:
            theta = super().forward(*args, **kwargs)
            self.cache = torch.stack([theta.float().sin(), theta.float().cos()])
        elif not self.training:  # We don't cache theta during evaluation to enable different batch sizes there
            theta = super().forward(*args, **kwargs)
            return torch.stack([theta.float().sin(), theta.float().cos()])
        return self.cache

    def apply_emb(self, x, theta):
        return ApplyRotaryEmbeddingInplacePrecomputed.apply(x, theta[0], theta[1], False)


def make_axial_pos_2d(h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None, relative_pos=True):
    if relative_pos:
        y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    else:
        y_min, y_max, x_min, x_max = -h / 2, h / 2, -w / 2, w / 2

    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)


# ============================================================ 1D RoPE


class AxialRoPE1D(AxialRoPEBase):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        learnable_freqs: bool = False,
        relative_canvas: bool = True,
        in_place: bool = False,
        half_embedding: bool = True,
    ):
        if half_embedding:
            assert dim % 2 == 0, "Half embedding is only supported for even dimensions"
            dim //= 2

        super().__init__(dim, n_heads, in_place=in_place)

        if learnable_freqs:
            assert not in_place, "In-place RoPE with learnable frequencies is not supported"

        self.learnable_freqs = learnable_freqs

        if relative_canvas:
            min_freq = math.pi
            max_freq = 10.0 * math.pi
        else:
            min_freq = 1 / 10_000
            max_freq = 1.0

        log_min = math.log(min_freq)
        log_max = math.log(max_freq)

        # Only one axis: single linspace
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 2 + 1)[:-1].exp()
        self.freqs = nn.Parameter(freqs.view(dim // 2, n_heads).T.contiguous(), requires_grad=learnable_freqs)

    def extra_repr(self):
        return f"dim={self.freqs.shape[-2] * 2}, n_heads={self.freqs.shape[-1]}"

    def forward(self, pos):
        # pos shape: (...,) or (..., 1)
        return pos[..., None] * self.freqs.to(pos.dtype)  # (..., n_heads, dim // 2)


def make_axial_pos_1d(length, relative_pos=True, align_corners=False, dtype=None, device=None):
    if relative_pos:
        x_min, x_max = -1.0, 1.0
    else:
        x_min, x_max = -length / 2, length / 2

    if align_corners:
        pos = torch.linspace(x_min, x_max, length, dtype=dtype, device=device)
    else:
        pos = centers(x_min, x_max, length, dtype=dtype, device=device)
    return pos.unsqueeze(1)  # shape: (length, 1)


# ============================================================ 3D RoPE


class AxialRoPE3D(AxialRoPEBase):
    def __init__(
        self,
        dim,
        n_heads,
        learnable_freqs=False,
        relative_canvas=True,
        use_log_freq=True,
        theta=100,
        half_embedding=True,
    ):
        if half_embedding:
            assert dim % 2 == 0, "Half embedding is only supported for even dimensions"
            dim //= 2
        super().__init__(dim, n_heads, in_place=False)
        self.learnable_freqs = learnable_freqs
        n_freqs = n_heads * dim // 8
        if relative_canvas:
            min_freq = math.pi
            max_freq = 10.0 * math.pi
        else:
            min_freq = 1 / 10_000
            max_freq = 1.0
        if use_log_freq:
            min_freq = math.log(min_freq)
            max_freq = math.log(max_freq)
        spatial_freqs = torch.linspace(min_freq, max_freq, n_freqs + 1)[:-1]
        if use_log_freq:
            spatial_freqs = spatial_freqs.exp()
        spatial_freqs = torch.stack([spatial_freqs] * 2)
        temporal_freqs = 1.0 / (theta ** (torch.arange(0, n_freqs).float() / (n_freqs)))
        self.spatial_freqs = nn.Parameter(
            spatial_freqs.view(2, dim // 8, n_heads).mT.contiguous(),
            requires_grad=learnable_freqs,
        )
        self.temporal_freqs = nn.Parameter(
            temporal_freqs.view(dim // 8, n_heads).T.contiguous(),
            requires_grad=learnable_freqs,
        )

    def extra_repr(self):
        return f"dim={self.spatial_freqs.shape[1] * 6}, n_heads={self.spatial_freqs.shape[0]}"

    def forward(self, pos):
        theta_t = pos[..., None, 0:1] * self.temporal_freqs
        theta_h = pos[..., None, 1:2] * self.spatial_freqs[0]
        theta_w = pos[..., None, 2:3] * self.spatial_freqs[1]
        result = torch.cat((theta_t, theta_h, theta_w), dim=-1)
        return result


def make_grid_3d(t_pos, h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(t_pos, h_pos, w_pos, indexing='ij'), dim=-1)
    t, h, w, d = grid.shape
    return grid.view(t * h * w, d)


def make_axial_pos_3d(t, h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None, relative_pos=True):
    if relative_pos:
        y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    else:
        y_min, y_max, x_min, x_max = -h / 2, h / 2, -w / 2, w / 2
    
    if align_corners:
        t_pos = torch.arange(t, dtype=dtype, device=device).float().to(dtype)
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        t_pos = torch.arange(t, dtype=dtype, device=device).float().to(dtype)
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid_3d(t_pos, h_pos, w_pos)
