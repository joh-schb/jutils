import os
import torch
import einops
import numpy as np
from PIL import Image
import torch.nn as nn
from torch import Tensor


def norm(im):
    return im / 127.5 - 1.


def denorm(im):
    return (im + 1.) * 127.5


def im2tensor(im, normalize_zero_one=False):
    """
    Args:
        im: PIL image or numpy array of shape (h, w, c) in range [0, 255]
        normalize_zero_one: If True, normalizes image to range [0, 1] instead of [-1, 1]
    Returns:
        Tensor of shape (3, h, w) normalized to either [-1, 1] or [0, 1]
    """
    if isinstance(im, Image.Image):
        im = np.array(im)
    assert len(im.shape) == 3, f"Image must be of shape (h, w, c). Got {im.shape}."
    if normalize_zero_one:
        im = im / 255.
    else:
        im = im / 127.5 - 1.
    im = einops.rearrange(im, 'h w c -> c h w')
    return torch.tensor(im)


def tensor2im(tensor, denormalize_zero_one=False):
    """
    Args:
        tensor: Tensor of shape (3, h, w)
        denormalize_zero_one: If True, denormalizes image from range [0, 1] otherwise
            from [-1, 1] to [0, 255]
    Returns:
        Numpy array of shape (h, w, 3) in range [0, 255]
    """
    assert len(tensor.shape) == 3, f"Tensor must be of shape (c, h, w). Got {tensor.shape}."
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    im = einops.rearrange(tensor, 'c h w -> h w c')
    if denormalize_zero_one:
        im = im * 255.
    else:
        im = (im + 1.) * 127.5
    im = np.clip(im, 0, 255).astype(np.uint8)
    return im


def alpha_compose(bg_im, fg_im, alpha=0.5):
    if not isinstance(bg_im, Image.Image):
        bg_im = Image.fromarray(bg_im)
    if not isinstance(fg_im, Image.Image):
        fg_im = Image.fromarray(fg_im)
        if fg_im.size != bg_im.size:
            fg_im = fg_im.resize(bg_im.size)
    image = bg_im.convert('RGB')
    fg = fg_im.convert('RGBA')
    alpha = int(255 * alpha)
    fg.putalpha(alpha)
    image.paste(fg, (0, 0), fg)
    return image


def get_original_reconstruction_image(x, x_hat, channels_first=False):
    """
    Returns pillow image of original and reconstruction images. Top row are originals, bottom
    row are reconstructions. Faster than creating a figure.

    Args:
        x: Original image of shape (n, h, w, c) or (h, w, c). If channels_first is true, the shape
            can also be (n, c, h, w) or (c, h, w).
        x_hat: Reconstructed image of same shape as x
        channels_first: If True, assumes x and x_hat are of shape (n, c, h, w) or (c, h, w)

    Returns:
        ims: Numpy array in shape [h, w, 3] with top row being originals and
            bottom row being reconstructions.
    """
    assert x.shape == x_hat.shape, f"Shapes must be equal. Got {x.shape} and {x_hat.shape}."
    # add batch dim if not present
    if len(x.shape) == 3:
        x = x[None]
        x_hat = x_hat[None]
    if channels_first:
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x_hat = einops.rearrange(x_hat, 'b c h w -> b h w c')

    x = einops.rearrange(x, 'b h w c -> h (b w) c')
    x_hat = einops.rearrange(x_hat, 'b h w c -> h (b w) c')
    ims = np.concatenate((x, x_hat), axis=0)

    return ims


def zero_pad(x, pad=2):
    """
    Pads image with zeros. If pad is an integer, pads with pad pixels on all sides.

    Args:
        x: Image of shape (..., h, w)
        pad: Number of pixels to pad on all sides or tuple of 4 ints (left, right, top, bottom)
    """
    is_torch = isinstance(x, torch.Tensor)
    x = torch.tensor(x) if not is_torch else x
    if isinstance(pad, tuple) or isinstance(pad, list):
        padding = pad
    else:
        padding = [pad, ] * 4
    x = torch.nn.functional.pad(x, padding)
    x = x.numpy() if not is_torch else x
    return x


def hwc2chw(im):
    return einops.rearrange(im, '... h w c -> ... c h w')


def chw2hwc(im):
    return einops.rearrange(im, '... c h w -> ... h w c')


def per_sample_min_max_normalization(x):
    """ Normalize each sample in a batch independently
    with min-max normalization to [0, 1] """
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)


def resize_ims(x: Tensor, size: int, mode: str = "bilinear", **kwargs):
    return nn.functional.interpolate(x, size=size, mode=mode, **kwargs)


def center_crop_np(im, new_height, new_width):
    assert (
        len(im.shape) == 3 or len(im.shape) == 2
    ), f"Image must be of shape (h, w, c) or (h, w). Got {im.shape}."
    height, width = im.shape[:2]
    left = (width - new_width)//2
    top = (height - new_height)//2
    right = (width + new_width)//2
    bottom = (height + new_height)//2
    im = im[top:bottom, left:right]
    return im


def center_crop_pil(im, new_height, new_width):
    width, height = im.size
    left = (width - new_width)//2
    top = (height - new_height)//2
    right = (width + new_width)//2
    bottom = (height + new_height)//2
    im = im.crop((left, top, right, bottom))
    return im


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    im_fp = os.path.join(cur_dir.split('jutils/vision')[0], 'assets', 'image.jpg')

    # alpha_compose(bg_im, fg_im, alpha=0.5)
    im_a = np.array(Image.open(im_fp))
    im_b = (np.random.randn(*im_a.shape) * 127.5 + 127.5).astype(np.uint8)
    im_c = alpha_compose(im_a, im_b, alpha=0.4)
    # im_c.show()

    # get_original_reconstruction_image(x, x_hat, channels_first=True)
    im_orig = np.array(Image.open(im_fp))
    im_orig = np.stack((im_orig, im_orig), axis=0)              # batched images
    im_rec = (np.random.randn(*im_orig.shape) * 127.5 + 127.5).astype(np.uint8)
    im_orig_rec = get_original_reconstruction_image(im_orig, im_rec)
    print("Shape of original vs reconstructed image:", im_orig_rec.shape)
    Image.fromarray(im_orig_rec).show()

    # img2tensor(img, normalize_zero_one=False)
    img = Image.open(im_fp)
    print("img2tensor(img).shape:", im2tensor(img).shape,
          f"min: {im2tensor(img).min()}, max: {im2tensor(img).max()}")

    # tensor2img(tensor, denormalize_zero_one=False)
    arr = im2tensor(img)
    print("tensor2img(tensor).shape:", tensor2im(arr).shape,
          f"min: {tensor2im(arr).min()}, max: {tensor2im(arr).max()}")

    # zero_pad(x, pad_size: int = 2)
    img = Image.open(im_fp)
    img = im2tensor(img, normalize_zero_one=True)
    print("zero_pad(x).shape:", zero_pad(img, (50, 100, 150, 200)).shape)
    Image.fromarray(
        tensor2im(zero_pad(img, (50, 100, 0, 200)), denormalize_zero_one=True)
    ).show()

    # hwc2chw(im)
    img = Image.open(im_fp)
    img = np.array(img)
    chw = hwc2chw(img)
    print("hwc2chw(im).shape:", hwc2chw(img).shape)
    print("chw2hwc(chw).shape:", chw2hwc(chw).shape)
    print("chw2hwc(chw[None].shape)", chw2hwc(chw[None]).shape)

    # per_sample_min_max_normalization(x)
    x = torch.arange(2 * 16).reshape(2, 1, 4, 4)
    print("per_sample_min_max_normalization(x):\n", per_sample_min_max_normalization(x))
    print("x:", x)

    # resize_ims(x: Tensor, size: int, mode: str = "bilinear", **kwargs)
    x = torch.arange(2 * 16).reshape(2, 1, 4, 4).float()
    print("resize_ims(x, size=8):\n", resize_ims(x, size=8))
    print("resize_ims(x, size=8, mode='nearest'):\n", resize_ims(x, size=8, mode='nearest'))

    # center_crop_np(im, new_height, new_width)
    img = Image.open(im_fp)
    img_arr = np.array(img)
    crop_pil = center_crop_pil(img, 128, 256)
    crop_np = center_crop_np(img_arr, 128, 256)
    both = np.concatenate((np.array(crop_pil), crop_np), axis=1)
    Image.fromarray(both).show()
    print("center_crop_np(img, 128, 256).shape:", crop_np.shape)
    print("center_crop_pil(img, 128, 256).size:", crop_pil.size)
