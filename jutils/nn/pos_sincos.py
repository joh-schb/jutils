import torch
import torch.nn as nn
from jaxtyping import Float
from collections.abc import Sequence

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


def _as_tuple(value: int | Sequence[int], ndim: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * ndim

    value = tuple(value)
    if len(value) != ndim:
        raise ValueError(f"{name} must have {ndim} entries, got {value}")
    return value


def sincos_pos_embed_1d(
    embed_dim: int,
    length: int,
    temperature: float = 10000.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a fixed axial 1D sine-cosine positional embedding with shape ``(length, dim)``."""
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    pos = torch.arange(length, dtype=torch.float64, device=device)
    omega = torch.arange(embed_dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (temperature ** (omega / (embed_dim / 2)))
    angles = pos.reshape(-1, 1) * omega.reshape(1, -1)
    pos_embed = torch.cat([angles.sin(), angles.cos()], dim=1)
    return pos_embed.to(dtype or torch.float32)


def sincos_pos_embed_2d(
    embed_dim: int,
    grid_size: int | Sequence[int],
    temperature: float = 10000.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a fixed axial 2D sine-cosine positional embedding with shape ``(h, w, dim)``."""
    h, w = _as_tuple(grid_size, 2, "grid_size")
    dim_h, dim_w = _pair_dims(embed_dim, 2)

    h_embed = sincos_pos_embed_1d(dim_h, h, temperature=temperature, dtype=dtype, device=device)
    w_embed = sincos_pos_embed_1d(dim_w, w, temperature=temperature, dtype=dtype, device=device)
    h_embed = h_embed[:, None, :].expand(h, w, dim_h)
    w_embed = w_embed[None, :, :].expand(h, w, dim_w)
    return torch.cat([h_embed, w_embed], dim=-1)


def sincos_pos_embed_3d(
    embed_dim: int,
    grid_size: int | Sequence[int],
    temperature: float = 10000.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a fixed axial 3D sine-cosine positional embedding with shape ``(t, h, w, dim)``."""
    t, h, w = _as_tuple(grid_size, 3, "grid_size")
    dim_t, dim_h, dim_w = _pair_dims(embed_dim, 3)

    t_embed = sincos_pos_embed_1d(dim_t, t, temperature=temperature, dtype=dtype, device=device)
    h_embed = sincos_pos_embed_1d(dim_h, h, temperature=temperature, dtype=dtype, device=device)
    w_embed = sincos_pos_embed_1d(dim_w, w, temperature=temperature, dtype=dtype, device=device)
    t_embed = t_embed[:, None, None, :].expand(t, h, w, dim_t)
    h_embed = h_embed[None, :, None, :].expand(t, h, w, dim_h)
    w_embed = w_embed[None, None, :, :].expand(t, h, w, dim_w)
    return torch.cat([t_embed, h_embed, w_embed], dim=-1)


def SincosPosEmbed1D(
    embed_dim: int,
    length: int,
    temperature: float = 10000.0,
    requires_grad: bool = False,
) -> Float[nn.Parameter, "length dim"]:
    """Return a sine-cosine ``nn.Parameter`` with shape ``(length, dim)``.
    Usage: ``self.pos_embed = SincosPosEmbed1D(...)``
    """
    pos_embed = sincos_pos_embed_1d(embed_dim, length, temperature=temperature)
    return nn.Parameter(pos_embed, requires_grad=requires_grad)


def SincosPosEmbed2D(
    embed_dim: int,
    grid_size: int | Sequence[int],
    temperature: float = 10000.0,
    requires_grad: bool = False,
) -> Float[nn.Parameter, "h w dim"]:
    """Return a sine-cosine ``nn.Parameter`` with shape ``(h, w, dim)``.
    Usage: ``self.pos_embed = SincosPosEmbed2D(...)``
    """
    pos_embed = sincos_pos_embed_2d(embed_dim, grid_size, temperature=temperature)
    return nn.Parameter(pos_embed, requires_grad=requires_grad)


def SincosPosEmbed3D(
    embed_dim: int,
    grid_size: int | Sequence[int],
    temperature: float = 10000.0,
    requires_grad: bool = False,
) -> Float[nn.Parameter, "t h w dim"]:
    """Return a sine-cosine ``nn.Parameter`` with shape ``(t, h, w, dim)``.
    Usage: ``self.pos_embed = SincosPosEmbed3D(...)``
    """
    pos_embed = sincos_pos_embed_3d(embed_dim, grid_size, temperature=temperature)
    return nn.Parameter(pos_embed, requires_grad=requires_grad)
