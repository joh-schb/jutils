import os
import einops
import numpy as np
from PIL import Image


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
