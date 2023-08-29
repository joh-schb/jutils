import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def save_as_gif(video, path, duration=120, loop=0, optimize=True):
    """
    Args:
        video: Video of type uint8 of shape (f, h, w, c)
        path: Filepath.
    """
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


if __name__ == "__main__":
    vid = np.random.randint(0, 255, (10, 256, 256, 3))

    # display video in jupyter notebook
    # display(animate_video(vid))

    # save video as gif
    save_as_gif(vid, 'test.gif')
