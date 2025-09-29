import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "RCPARAMS",
    "ALL_RCPARAMS",
    "set_rcparams",
    "extract_image_from_fig",
]
# ===============================================================================================


RCPARAMS = {
    # figure
    'figure.dpi': 300,
    'figure.figsize': (6, 2.5),
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.75,
    'savefig.bbox': 'tight',
    # grid
    'axes.grid': True,
    'grid.linewidth': 0.5,
    'axes.grid.axis': 'both',       # 'x', 'y', or 'both'
    'grid.color': 'lightgray',
    # linestyle
    'lines.marker': 'o',
    'lines.markeredgewidth': 1.25,
    'lines.markeredgecolor': 'auto',   # 'auto' or 'white'
    'lines.markersize': 6,              # 6 or 8
    'lines.linewidth': 2,
    # x-ticks
    'xtick.major.size': 3.75,
    'xtick.major.width': 0.75,
    'xtick.minor.size': 2.25,
    'xtick.minor.width': 0.375,
    'xtick.minor.visible': False,       # small ticks
    # y-ticks
    'ytick.major.size': 3.75,
    'ytick.major.width': 0.75,
    'ytick.minor.size': 2.25,
    'ytick.minor.width': 0.375,
    'ytick.minor.visible': False,       # small ticks
    # font
    'font.weight': 300,
    'font.size': 10,
    'font.family': 'sans-serif',        # ranking: 'Avenir', 'Palatino', 'PT Serif', 'Times New Roman', 'Helvetica'
}
ALL_RCPARAMS = dict(plt.rcParams)


def set_rcparams(fontfamily='sans-serif', fontsize=10, xyticks_minor=False, grid=True, figsize=(6, 2.5), reset=True):
    """
    Args:
        fontfamily: 'Avenir', 'Palatino', 'PT Serif', 'Times New Roman', 'Helvetica', or default 'sans-serif'
        fontsize: size of the output font
        xyticks_minor: show small ticks (x, y)
        grid: show grid
        figsize: size of the figure
    """
    if reset:
        plt.rcdefaults()
    new_rcparams = ALL_RCPARAMS.copy()
    new_rcparams['xtick.minor.visible'] = xyticks_minor
    new_rcparams['ytick.minor.visible'] = xyticks_minor
    new_rcparams['font.size'] = fontsize
    new_rcparams['font.family'] = fontfamily
    new_rcparams['figure.figsize'] = figsize
    new_rcparams['axes.grid'] = grid
    plt.rcParams.update(new_rcparams)


def extract_image_from_fig(fig: plt.Figure) -> np.ndarray:
    """ Return an RGB image array from a Matplotlib Figure (h, w, 3) uint8. """
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return img


if __name__ == "__main__":
    x = np.linspace(0, 4, 10)
    
    fig, axes = plt.subplots(1, 1)
    plt.title("RCParams")
    plt.xlabel("x")
    plt.ylabel("$\delta_1 \\uparrow$")
    for i in range(5):
        y = -np.log(x + 0.001) + i * 3
        axes.plot(x, y, '-o', label=f'Color {i}')
    plt.legend()
    plt.savefig("_rcparams.png")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    img = extract_image_from_fig(fig)
    print("Extracted image shape:", img.shape)
