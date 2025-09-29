import os
import torch
import einops
import textwrap
import numpy as np
from PIL import Image
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont


__all__ = [
    "norm",
    "denorm",
    "im2tensor",
    "tensor2im",
    "alpha_compose",
    "alpha_compose_heatmap",
    "get_original_reconstruction_image",
    "zero_pad",
    "hwc2chw",
    "chw2hwc",
    "per_sample_min_max_normalization",
    "resize_ims",
    "center_crop_np",
    "center_crop_pil",
    "resize_shorter_side_pil",
    "ims_to_grid",
    "soft_wrap",
    "text_to_canvas",
]
# ===============================================================================================


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
    return torch.tensor(im).float()


def tensor2im(tensor, denormalize_zero_one=False):
    """
    Args:
        tensor: Tensor of shape (..., 3, h, w) in range [-1, 1] or [0, 1]
        denormalize_zero_one: If True, denormalizes image from range [0, 1] otherwise
            from [-1, 1] to [0, 255]
    Returns:
        Numpy array of shape (h, w, 3) in range [0, 255] if tensor shape is (3, h, w)
            or (1, 3, h, w). Otherwise, returns array of shape (..., h, w, 3).
    """
    if len(tensor.shape) == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    im = einops.rearrange(tensor, '... c h w -> ... h w c')
    if denormalize_zero_one:
        im = im * 255.
    else:
        im = (im + 1.) * 127.5
    im = np.clip(im, 0, 255).astype(np.uint8)
    return im


def alpha_compose(bg_im, fg_im, alpha=0.5, resizer=Image.Resampling.BILINEAR):
    if not isinstance(bg_im, Image.Image):
        bg_im = Image.fromarray(bg_im)
    if not isinstance(fg_im, Image.Image):
        fg_im = Image.fromarray(fg_im)
    if fg_im.size != bg_im.size:
        fg_im = fg_im.resize(bg_im.size, resample=resizer)
    image = bg_im.convert('RGB')
    fg = fg_im.convert('RGBA')
    alpha = int(255 * alpha)
    fg.putalpha(alpha)
    image.paste(fg, (0, 0), fg)
    return image


def alpha_compose_heatmap(image, heatmap, alpha=0.5, cmap='viridis', resizer=Image.Resampling.BILINEAR):
    """
    Args:
        image: PIL image
        heatmap: torch.Tensor or np.ndarray of shape (h, w)
        alpha: float, alpha value for the heatmap
        cmap: str, pyplot colormap for the heatmap (default: 'viridis')
        resizer: PIL resampling filter for resizing the heatmap (default: Image.Resampling.BILINEAR)
    
    Returns:
        PIL image
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    assert heatmap.ndim == 2, 'heatmap must be 2D (H, W)'

    # min-max normalize the heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    cmapper = plt.get_cmap(cmap)
    heatmap = cmapper(heatmap, bytes=True)[:, :, :3]
    heatmap = Image.fromarray(heatmap)

    return alpha_compose(image, heatmap, alpha=alpha, resizer=resizer)


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


def resize_shorter_side_pil(image: Image, target_size: int):
    """
    Args:
        image: PIL image or numpy array of shape (h, w, c)
        target_size: Size of the shorter side
    Returns:
        PIL image or numpy array of shape (h', w', c)
    """
    is_np = isinstance(image, np.ndarray)
    if is_np:
        assert len(image.shape) == 3, f"Image must be of shape (h, w, c). Got {image.shape}."
        image = Image.fromarray(image)
    w, h = image.size  # PIL uses (w, h)

    if h < w:
        new_h, new_w = target_size, int(w * (target_size / h))
    else:
        new_w, new_h = target_size, int(h * (target_size / w))

    resized_pil = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if is_np:
        return np.array(resized_pil)
    return resized_pil


def ims_to_grid(ims, stack="row", split=4, channel_last=False):
    """
    Args:
        ims: Tensor of shape (b, c, h, w)
        stack: "row" or "col"
        split: If 'row' stack by rows, if 'col' stack by columns.
    Returns:
        Tensor of shape (h, w, c)
    """
    if stack not in ["row", "col"]:
        raise ValueError(f"Unknown stack type {stack}")
    from_ = 'h w c' if channel_last else 'c h w'
    if split is not None and ims.shape[0] % split == 0:
        splitter = dict(b1=split) if stack == "row" else dict(b2=split)
        ims = einops.rearrange(ims, f"(b1 b2) {from_} -> (b1 h) (b2 w) c", **splitter)
    else:
        to = "(b h) w c" if stack == "row" else "h (b w) c"
        ims = einops.rearrange(ims, f"b {from_} -> " + to)
    return ims


def soft_wrap(s: str, n: int) -> str:
    return textwrap.fill(s, width=n, break_long_words=True, break_on_hyphens=False)


def text_to_canvas(txt, h, w=None, background=(0, 0, 0), fontcolor=(255, 255, 0), font_size=24):
    """
    Create an image with text and return it as a NumPy array of shape (h, w, 3) dtype uint8.
    """
    w = w or h
    image = Image.new("RGB", (w, h), color=background)
    draw = ImageDraw.Draw(image)

    try: font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError: font = ImageFont.load_default()

    # Calculate text size and position it in the center
    text_bbox = draw.textbbox((0, 0), txt, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((w - text_width) // 2, (h - text_height) // 2)

    draw.text(position, txt, fill=fontcolor, font=font)
    
    return np.array(image)


""" Unit Testing """


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    im_fp = os.path.join(cur_dir.split('jutils/vision')[0], 'assets', 'image.jpg')

    # alpha_compose(bg_im, fg_im, alpha=0.5)
    im_a = np.array(Image.open(im_fp))
    im_b = (np.random.randn(*im_a.shape) * 127.5 + 127.5).astype(np.uint8)
    im_c = alpha_compose(im_a, im_b, alpha=0.4)
    # im_c.show()

    # alpha_compose_heatmap(image, heatmap, alpha=0.5, cmap='viridis', resizer=Image.Resampling.BILINEAR)
    im = Image.open(im_fp)
    x = np.linspace(-1, 1, 100)
    x, y = np.meshgrid(x, x)
    gaussian = np.exp(-(x**2 + y**2) / (2 * 0.5**2))
    im_composed = alpha_compose_heatmap(im, gaussian, alpha=0.5, cmap='viridis')
    im_composed.show()

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

    # ims_to_grid(ims, stack="row", split=4)
    ims = im2tensor(Image.open(im_fp).resize((128, 128))).unsqueeze(0).repeat(16, 1, 1, 1)
    grid1 = ims_to_grid(ims, stack="row", split=4)
    grid2 = ims_to_grid(ims.permute(0, 2, 3, 1), stack="col", split=2, channel_last=True)
    print("ims_to_grid(ims, stack='row', split=4).shape:", grid1.shape)
    print("ims_to_grid(ims, stack='col', split=2, channel_last=True).shape:", grid2.shape)
    Image.fromarray(denorm(grid1).to(torch.uint8).numpy()).show()

    # resize_shorter_side_pil(image: Image, target_size: int)
    img = Image.open(im_fp)
    resized = resize_shorter_side_pil(img, 80)
    print("resize_shorter_side_pil(img, 256).size:", resized.size)
    resized.show()

    # text_to_canvas(txt, h, w=None, background=(0, 0, 0), fontcolor=(255, 255, 0), font_size=24)
    txt = "Hello, World! How are you? This is a long text that should wrap"
    canvas = text_to_canvas(soft_wrap(txt, 20), 100, 200, background=(0, 0, 0), fontcolor=(255, 255, 0), font_size=24)
    print("text_to_canvas(txt, h, w).shape:", canvas.shape)
    Image.fromarray(canvas).show()
