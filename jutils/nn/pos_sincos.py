import torch
import torch.nn as nn
from jaxtyping import Float

__all__ = [
    "sincos_pos_embed_1d",
    "sincos_pos_embed_2d",
    "sincos_pos_embed_3d",
    "SincosPosEmbed1D",
    "SincosPosEmbed2D",
    "SincosPosEmbed3D",
]
# ===================================================================================================


def _pair_dims(total_dim: int, ndim: int) -> list[int]:
    if total_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {total_dim}")

    n_pairs = total_dim // 2
    if n_pairs < ndim:
        raise ValueError(f"embed_dim={total_dim} is too small for {ndim} axes")

    base = n_pairs // ndim
    remainder = n_pairs % ndim
    return [2 * (base + (axis < remainder)) for axis in range(ndim)]


def sincos_pos_embed_1d(
    length: int,
    dim: int,
    temperature: float = 10000.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a fixed axial 1D sine-cosine positional embedding with shape ``(length, dim)``."""
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    pos = torch.arange(length, dtype=torch.float64, device=device)
    omega = torch.arange(dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (temperature ** (omega / (dim / 2)))
    angles = pos.reshape(-1, 1) * omega.reshape(1, -1)
    pos_embed = torch.cat([angles.sin(), angles.cos()], dim=1)
    return pos_embed.to(dtype or torch.float32)


def sincos_pos_embed_2d(
    h: int,
    w: int,
    dim: int,
    temperature: float = 10000.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a fixed axial 2D sine-cosine positional embedding with shape ``(h, w, dim)``."""
    dim_h, dim_w = _pair_dims(dim, 2)

    h_embed = sincos_pos_embed_1d(h, dim_h, temperature=temperature, dtype=dtype, device=device)
    w_embed = sincos_pos_embed_1d(w, dim_w, temperature=temperature, dtype=dtype, device=device)
    h_embed = h_embed[:, None, :].expand(h, w, dim_h)
    w_embed = w_embed[None, :, :].expand(h, w, dim_w)
    return torch.cat([h_embed, w_embed], dim=-1)


def sincos_pos_embed_3d(
    t: int,
    h: int,
    w: int,
    dim: int,
    temperature: float = 10000.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a fixed axial 3D sine-cosine positional embedding with shape ``(t, h, w, dim)``."""
    dim_t, dim_h, dim_w = _pair_dims(dim, 3)

    t_embed = sincos_pos_embed_1d(t, dim_t, temperature=temperature, dtype=dtype, device=device)
    h_embed = sincos_pos_embed_1d(h, dim_h, temperature=temperature, dtype=dtype, device=device)
    w_embed = sincos_pos_embed_1d(w, dim_w, temperature=temperature, dtype=dtype, device=device)
    t_embed = t_embed[:, None, None, :].expand(t, h, w, dim_t)
    h_embed = h_embed[None, :, None, :].expand(t, h, w, dim_h)
    w_embed = w_embed[None, None, :, :].expand(t, h, w, dim_w)
    return torch.cat([t_embed, h_embed, w_embed], dim=-1)


def SincosPosEmbed1D(
    length: int,
    dim: int,
    temperature: float = 10000.0,
    requires_grad: bool = False,
) -> Float[nn.Parameter, "length dim"]:
    """Return a 1D sine-cosine ``nn.Parameter`` for ``length`` positions and ``dim`` features.

    The returned parameter has shape ``(length, dim)`` and is frozen by default.
    Usage: ``self.pos_embed = SincosPosEmbed1D(length, dim)``.
    """
    pos_embed = sincos_pos_embed_1d(length, dim, temperature=temperature)
    return nn.Parameter(pos_embed, requires_grad=requires_grad)


def SincosPosEmbed2D(
    h: int,
    w: int,
    dim: int,
    temperature: float = 10000.0,
    requires_grad: bool = False,
) -> Float[nn.Parameter, "h w dim"]:
    """Return a 2D sine-cosine ``nn.Parameter`` for an ``(h, w)`` grid and ``dim`` features.

    The returned parameter has shape ``(h, w, dim)`` and is frozen by default.
    Usage: ``self.pos_embed = SincosPosEmbed2D(h, w, dim)``.
    """
    pos_embed = sincos_pos_embed_2d(h, w, dim, temperature=temperature)
    return nn.Parameter(pos_embed, requires_grad=requires_grad)


def SincosPosEmbed3D(
    t: int,
    h: int,
    w: int,
    dim: int,
    temperature: float = 10000.0,
    requires_grad: bool = False,
) -> Float[nn.Parameter, "t h w dim"]:
    """Return a 3D sine-cosine ``nn.Parameter`` for a ``(t, h, w)`` grid and ``dim`` features.

    The returned parameter has shape ``(t, h, w, dim)`` and is frozen by default.
    Usage: ``self.pos_embed = SincosPosEmbed3D(t, h, w, dim)``.
    """
    pos_embed = sincos_pos_embed_3d(t, h, w, dim, temperature=temperature)
    return nn.Parameter(pos_embed, requires_grad=requires_grad)
