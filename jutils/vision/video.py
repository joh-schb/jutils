import torch
import einops
import numpy as np
from PIL import Image
from typing import Union, Tuple
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation


def vid2tensor(video, normalize_zero_one=False):
    """
    Args:
        video: Numpy array of shape (T, H, W, C) or (B, T, H, W, C), range [0, 255]
        normalize_zero_one: If True, normalize to [0, 1]; otherwise to [-1, 1]
    Returns:
        Tensor of shape (C, T, H, W) or (B, C, T, H, W), normalized
    """
    assert len(video.shape) in {4, 5}, f"Expected shape (T, H, W, C) or (B, T, H, W, C), got {video.shape}"
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video).float()
    video = video / 255. if normalize_zero_one else video / 127.5 - 1.
    video = einops.rearrange(video, '... t h w c -> ... c t h w')
    return video


def tensor2vid(tensor, denormalize_zero_one=False):
    """
    Args:
        tensor: Tensor of shape (B, C, T, H, W) or (C, T, H, W), with values in [-1, 1] or [0, 1]
        denormalize_zero_one: If True, assumes input is in [0, 1]; otherwise in [-1, 1]
    Returns:
        Numpy array of shape (B, T, H, W, C) or (T, H, W, C) in range [0, 255], uint8.
    """
    assert len(tensor.shape) in {4, 5}, f"Expected shape (B, C, T, H, W) or (C, T, H, W), got {tensor.shape}"
    if len(tensor.shape) == 5 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    video = einops.rearrange(tensor, '... c t h w -> ... t h w c')
    video = video * 255. if denormalize_zero_one else (video + 1.) * 127.5
    video = np.clip(video, 0, 255).astype(np.uint8)
    return video


def center_crop_video(video, crop_size: Union[int, Tuple[int, int]]):
    """
    Args:
        video (np.ndarray or torch.Tensor): Input video of shape (B, C, T, H, W) or (C, T, H, W)
        crop_size (tuple or int): Desired height and/or width

    Returns:
        Cropped video of same type and dimensionality
    """
    assert len(video.shape) in {4, 5}, f"Input video must be of shape (C, T, H, W) or (B, C, T, H, W), got {video.shape}"
    *_, h, w = video.shape
    if isinstance(crop_size, int):
        crop_h, crop_w = crop_size, crop_size
    elif isinstance(crop_size, tuple) and len(crop_size) == 2:
        crop_h, crop_w = crop_size
    else:
        raise ValueError("crop_size must be an int or a tuple of two ints (height, width)")
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    cropped = video[..., top:top+crop_h, left:left+crop_w]
    return cropped


def save_as_gif(video, path, fps=15, loop=0, optimize=True):
    """
    Args:
        video: Video of type uint8 of shape (f, h, w, c)
        path: Filepath.
    """
    duration = int(1000 / fps)
    if not isinstance(video, np.ndarray):
        video = np.array(video)
    if not video.dtype == np.uint8:
        video = video.astype(np.uint8)
    images = [Image.fromarray(frame) for frame in video]
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)


def animate_video(frames, fps=5, return_html=False):
    """
    Args:
        frames: Video of shape (f, h, w, c)
    """
    def animate(frames, fps):
        if not isinstance(frames, np.ndarray):
            frames = np.array(frames)
        num_frames, height, width, channels = frames.shape
        # Create a figure and axis for the animation
        fig, ax = plt.subplots()
        ax.axis('off')
        img = ax.imshow(frames[0])

        # Update function for animation
        def update(frame):
            img.set_array(frame)
            return img,

        # Create the animation
        animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=True)
        # Display the animation
        plt.close(animation._fig)
        return animation.to_jshtml()

    if return_html:
        return HTML(animate(frames, fps))
    return animate(frames, fps)


def colorize_border(x, cond_frames: int, channel=0, value=255, pad=1):
    """
    Args:
        x: Input of shape (b, c, f, h, w) in uint8 and range [0, 255].
        cond_frames: Number of conditioning frames. Pads first n frames.
        channel: Color channel (0=red, 1=green, 2=blue).
        value: Value (should be either 1 or 255).
        pad: Padding size (default=1).
    Return:
        x: Input with colored border and shape (b, c, f, h + 2*pad, w + 2*pad).
    """
    if pad > 0:
        is_torch = isinstance(x, torch.Tensor)
        x = torch.tensor(x) if not is_torch else x
        x = torch.nn.functional.pad(x, [pad, ] * 4)
        x = x.numpy() if not is_torch else x
    else:
        pad = 1
    x[:, channel, :cond_frames, :pad, :] = value      # top
    x[:, channel, :cond_frames, -pad:, :] = value     # bottom
    x[:, channel, :cond_frames, :, :pad] = value      # left
    x[:, channel, :cond_frames, :, -pad:] = value     # right
    return x


if __name__ == "__main__":
    vid = np.random.randint(0, 256, (10, 128, 128, 3), dtype=np.uint8)

    # vid2tensor and tensor2vid
    vid_out = tensor2vid(vid2tensor(vid))
    print("tensor2vid(vid2tensor(video)) - close:", np.allclose(vid.astype(np.float32), vid_out.astype(np.float32), atol=1))

    # center_crop_video
    crop_vid = torch.randn((3, 10, 128, 128))
    cropped_vid = center_crop_video(crop_vid, (64, 100))
    print(f"center_crop_video shape: {crop_vid.shape} -> {cropped_vid.shape}")

    # display video in jupyter notebook
    # display(animate_video(vid))

    # save video as gif
    save_as_gif(vid, '_test.gif')

    # colorize_border(x, cond_frames, channel=0, value=255, pad=1)
    vid = np.random.randint(0, 255, (1, 3, 10, 256, 256))
    vid = colorize_border(vid, 5, channel=0, value=255, pad=20)[0]
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    save_as_gif(vid, '_colorized-border.gif')
