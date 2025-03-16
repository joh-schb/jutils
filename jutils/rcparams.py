import matplotlib.pyplot as plt

plt.rcdefaults()
RCPARAMS = {
    # figure
    'figure.dpi': 150,
    'figure.figsize': (6, 2.5),
    'savefig.pad_inches': 0.05,
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
    'font.family': 'Avenir',        # ranking: 'Avenir', 'Palatino', 'PT Serif', 'Times New Roman', 'Helvetica'
}
plt.rcParams.update(RCPARAMS)
ALL_RCPARAMS = dict(plt.rcParams)


def set_rcparams():
    plt.rcdefaults()
    plt.rcParams.update(RCPARAMS)


if __name__ == "__main__":
    import numpy as np
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
