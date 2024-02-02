import torch
import einops
import numpy as np
import matplotlib.pyplot as plt


def exists(v):
    return v is not None


def percentile_per_sample(x, percentile):
    return np.percentile(x, percentile, axis=[*range(1, x.ndim)])


def pad_vector_like_x(v, x):
    """
    Function to reshape the vector by the number of dimensions    of x. E.g. x (bs, c, h, w), v (bs) -> v (bs, 1, 1, 1).
    Args:        x : Tensor, shape (bs, *dim)        v : FloatTensor, shape (bs)
    Returns:        vec : Tensor, shape (bs, number of x dimensions)    """
    if isinstance(v, float):
        return v
    return v.reshape(-1, *([1] * (x.ndim - 1)))


def per_sample_min_max_normalization(x):
    """ Normalize each sample in a batch independently
    with min-max normalization to [0, 1] """
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)


def colorize_depth_map(
        depth,
        vmin=None,
        vmax=None,
        percentiles=False,
        cmap="Spectral",
        invalid_mask=None,
        invalid_color=(0, 0, 0),
        inverse=False
):
    """
    Colorize a depth map using a matplotlib colormap. It actually
    converts the depths to the inverse, so that closer objects
    are brighter.

    Args:
        depth: Depth tensor of shape (b, 1, c, w) or (b, h, w) with
            planar depth values ranging from 0 to inf.
        vmin: Minimum depth value to use for scaling the colormap. Can
            also be a percentile value if percentiles is True. If None,
            values in the batch are not min-clipped.
        vmax: Maximum depth value to use for scaling the colormap. Can
            also be a percentile value if percentiles is True. If None,
            values in the batch are not max-clipped.
        percentiles: If True, vmin and vmax are interpreted as percentiles
            of the depth values in the batch (per sample!).
        cmap: Name of the matplotlib colormap to use.
        invalid_mask: Boolean mask of shape (b, h, w) that is True where
            the depth values are invalid.
        invalid_color: RGB color to use for invalid depth values.
        inverse: If True, the depth values are inverted before colorization.
    """
    if len(depth.shape) == 4:
        assert depth.shape[1] == 1, "Depth must have 1 channel."
        depth = depth.squeeze(1)
    assert len(depth.shape) == 3, "Depth must have shape (b, h, w) or (b, 1, h, w)."
    assert depth.min() >= 0, "Depth must be non-negative."

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    # clip values with vmin and vmax
    if vmin is not None and percentiles:
        assert 0 <= vmin < 100, "vmin must be in [0, 100] if using percentiles"
        vmin = percentile_per_sample(depth, vmin)
        vmin = pad_vector_like_x(vmin, depth)
    if vmax is not None and percentiles:
        assert 0 < vmax <= 100, "vmax must be in [0, 100] if using percentiles"
        vmax = percentile_per_sample(depth, vmax)
        vmax = pad_vector_like_x(vmax, depth)
    if exists(vmin) or exists(vmax):
        # clip values between vmin and vmax
        depth = np.clip(depth, vmin, vmax)

    # take inverse of depth
    if inverse:
        depth = 1.0 / depth

    # normalize to [0, 1]
    depth = per_sample_min_max_normalization(depth)

    # apply colormap
    cmapper = plt.get_cmap(cmap)
    depth = cmapper(depth, bytes=True)[..., :3]         # (b, h, w, 3)

    if invalid_mask is not None:
        depth[invalid_mask] = invalid_color

    return depth


if __name__ == "__main__":
    from PIL import Image
    depthy = torch.arange(2*100*100).reshape(2, 1, 100, 100).float()

    colorized = colorize_depth_map(
        depthy,
        vmin=2,
        vmax=98,
        percentiles=True,
        cmap="Spectral",
        inverse=False
    )
    print("colorized.shape:", colorized.shape)
    print("colorized.min():", colorized.min())
    print("colorized.max():", colorized.max())
    print("colorized.dtype:", colorized.dtype)
    Image.fromarray(colorized[0]).save("colorized-depth-map.png")
