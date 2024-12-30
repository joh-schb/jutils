import einops
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


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
    vid = np.random.randint(0, 255, (10, 256, 256, 3))

    # display video in jupyter notebook
    # display(animate_video(vid))

    # save video as gif
    save_as_gif(vid, 'test.gif')

    # colorize_border(x, cond_frames, channel=0, value=255, pad=1)
    vid = np.random.randint(0, 255, (1, 3, 10, 256, 256))
    vid = colorize_border(vid, 5, channel=0, value=255, pad=20)[0]
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    save_as_gif(vid, 'colorized-border.gif')
