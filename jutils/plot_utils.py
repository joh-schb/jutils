import numpy as np
from pathlib import Path
from importlib import resources
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

__all__ = [
    "RCPARAMS",
    "ALL_RCPARAMS",
    "set_rcparams",
    "extract_image_from_fig",
    "latex_pt_to_inch",
    "inch_to_latex_pt",
    "make_row_of_axes",
    "round_bar_tops",
    "label_bars",
    "register_google_fonts",
    "set_font_family",
]
# ===============================================================================================


POINTS_PER_INCH = 72.27  # TeX points per inch


RCPARAMS = {
    # figure
    "figure.dpi": 300,
    "figure.figsize": (6, 2.5),
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.75,
    "savefig.bbox": "tight",
    # grid
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "axes.grid.axis": "both",  # 'x', 'y', or 'both'
    "grid.color": "lightgray",
    # linestyle
    "lines.marker": "o",
    "lines.markeredgewidth": 1.25,
    "lines.markeredgecolor": "auto",  # 'auto' or 'white'
    "lines.markersize": 6,  # 6 or 8
    "lines.linewidth": 2,
    # x-ticks
    "xtick.major.size": 3.75,
    "xtick.major.width": 0.75,
    "xtick.minor.size": 2.25,
    "xtick.minor.width": 0.375,
    "xtick.minor.visible": False,  # small ticks
    # y-ticks
    "ytick.major.size": 3.75,
    "ytick.major.width": 0.75,
    "ytick.minor.size": 2.25,
    "ytick.minor.width": 0.375,
    "ytick.minor.visible": False,  # small ticks
    # font
    "font.weight": 300,
    "font.size": 10,
    "font.family": "sans-serif",  # ranking: 'Avenir', 'Palatino', 'PT Serif', 'Times New Roman', 'Helvetica'
}
ALL_RCPARAMS = dict(plt.rcParams)


def set_rcparams(fontfamily="sans-serif", fontsize=10, xyticks_minor=False, grid=True, figsize=(6, 2.5), reset=True):
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
    new_rcparams["xtick.minor.visible"] = xyticks_minor
    new_rcparams["ytick.minor.visible"] = xyticks_minor
    new_rcparams["font.size"] = fontsize
    new_rcparams["font.family"] = fontfamily
    new_rcparams["figure.figsize"] = figsize
    new_rcparams["axes.grid"] = grid
    plt.rcParams.update(new_rcparams)


def extract_image_from_fig(fig: plt.Figure) -> np.ndarray:
    """Return an RGB image array from a Matplotlib Figure (h, w, 3) uint8."""
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return img


def latex_pt_to_inch(pt):
    return pt / POINTS_PER_INCH


def inch_to_latex_pt(inch):
    return inch * POINTS_PER_INCH


def round_bar_tops(ax, bars, radius: float = 6):
    """Round vertical bar tips with a pixel radius; call after setting limits and layout.

    bars = ax.bar(["A", "B"], [72.4, 81.2])
    round_bar_tops(ax, bars, radius=10)
    """
    ax.figure.canvas.draw()
    for bar in bars:
        (x0, y0), (x1, y1) = ax.transData.transform(bar.get_bbox().get_points())
        x0, x1 = sorted((x0, x1))
        r = max(0, min(radius, (x1 - x0) / 2, abs(y1 - y0) / 2))
        dy = np.copysign(r, y1 - y0)
        vertices = [
            (x0, y0),
            (x1, y0),
            (x1, y1 - dy),
            (x1, y1),
            (x1 - r, y1),
            (x0 + r, y1),
            (x0, y1),
            (x0, y1 - dy),
            (x0, y0),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.CLOSEPOLY,
        ]
        patch = PathPatch(MplPath(ax.transData.inverted().transform(vertices), codes))
        patch.update_from(bar)
        patch.set_transform(ax.transData)
        bar.remove()
        ax.add_patch(patch)


def label_bars(ax, bars, fmt: str = "%.1f", padding: float = 3, **kwargs):
    """Label bar tips, with padding in points.

    ```
    bars = ax.bar(["A", "B"], [72.4, 81.2])
    label_bars(ax, bars, fmt="%.1f", padding=4)     # or: labels=["Base", "Ours"]
    """
    return ax.bar_label(bars, fmt=fmt, padding=padding, **kwargs)


def make_row_of_axes(
    width_pt: float,
    ratios: list[float] | list[int],
    aspect_ratio: float = 1.0,  # height / width per axis (based on widest axis)
    gap_pts: float | list[float] = 6.0,  # single gap (uniform) OR list of length n-1
    left_pt: float = 6.0,
    right_pt: float = 6.0,
):
    """
    Create a 1xN row of axes that fits a LaTeX text width with precise margins and
    either uniform or per-gap spacing (all in TeX points).

    Find specs in LaTeX document with:
    Text width: \the\textwidth \par
    ColSep: \the\columnsep \par
    Font: \expandafter\string\the\font \par

    Args:
        width_pt: Total width to occupy (pt), e.g., \\the\\textwidth in pt.
        ratios: Relative widths for the N axes, e.g., [2, 1, 1].
        aspect_ratio: Height/width used to compute figure height from the widest axis.
        gap_pts: Single float for uniform gaps OR list of length N-1 for per-gap control (pt).
        left_pt, right_pt: figure margins (pt).

    Returns:
        (fig, axes) where axes is a Python list of length N.
    """
    n = len(ratios)
    if n < 1:
        raise ValueError("ratios must contain at least one entry.")
    ratios = [float(r) for r in ratios]
    total_ratio = float(sum(ratios))
    if total_ratio <= 0:
        raise ValueError("Sum of ratios must be positive.")

    # Normalize gap list
    if isinstance(gap_pts, (int, float)):
        gap_list = [float(gap_pts)] * max(n - 1, 0)
    else:
        gap_list = [float(g) for g in gap_pts]
        if len(gap_list) != n - 1:
            raise ValueError(f"gap_pts must be a single number or a list of length {n-1}.")

    # Compute available content width (excluding margins and gaps)
    gaps_total_pt = sum(gap_list)
    content_pt = width_pt - left_pt - right_pt - gaps_total_pt
    if content_pt <= 0:
        raise ValueError("Non-positive content width. Reduce gaps/margins or increase text_width_pt.")

    # Axis widths (pt) according to ratios
    axis_widths_pt = [content_pt * r / total_ratio for r in ratios]

    # Figure size in inches
    fig_w_in = width_pt / POINTS_PER_INCH
    axis_w_max_in = max(axis_widths_pt) / POINTS_PER_INCH
    fig_h_in = axis_w_max_in * aspect_ratio

    # Build interleaved width ratios for GridSpec: [ax1, gap1, ax2, gap2, ..., axN]
    width_ratios_pt = []
    for i in range(n):
        width_ratios_pt.append(axis_widths_pt[i])
        if i < n - 1:
            width_ratios_pt.append(gap_list[i])

    # Create figure and gridspec with explicit spacer columns (so wspace=0)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    gs = fig.add_gridspec(1, 2 * n - 1, width_ratios=width_ratios_pt, wspace=0.0, hspace=0.0)

    # Margins as fractions of figure size
    left = left_pt / width_pt
    right = 1.0 - (right_pt / width_pt)
    fig.subplots_adjust(left=left, right=right)

    # Create axes in axis columns (even indices 0,2,4,...)
    axes = [fig.add_subplot(gs[0, 2 * i]) for i in range(n)]

    return fig, axes


# ===================================================================================================


def _iter_ttf_files(package_subdir: str):
    """
    Yield all .ttf files under a given importlib.resources package subdir.

    Works on older Python versions where Traversable does not implement .rglob().
    """
    base = resources.files(package_subdir)

    def _walk(trav):
        for child in trav.iterdir():
            if child.is_file() and child.name.lower().endswith(".ttf"):
                yield child
            elif child.is_dir():
                yield from _walk(child)

    yield from _walk(base)


def _register_all_ttf_under(package_subdir: str):
    """
    Register all .ttf fonts under a given package subdirectory.
    package_subdir is something like "jutils.fonts.Fira_Sans".
    """
    registered = set()

    for ttf_path in _iter_ttf_files(package_subdir):
        ttf_path = Path(ttf_path)
        fm.fontManager.addfont(str(ttf_path))
        props = fm.FontProperties(fname=str(ttf_path))
        registered.add(props.get_name())

    return registered


def register_google_fonts():
    """
    Register all TTFs from jutils/fonts/Fira_Sans and jutils/fonts/Montserrat.

    Returns a set of all discovered font family names.
    """
    families = set()
    families |= _register_all_ttf_under("jutils.fonts.Fira_Sans")
    families |= _register_all_ttf_under("jutils.fonts.Montserrat")
    families |= _register_all_ttf_under("jutils.fonts.Quicksand")
    return families


def set_font_family(family: str):
    """
    Set the Matplotlib default font family to the given name, e.g. "Fira Sans".
    """
    plt.rcParams["font.family"] = family


if __name__ == "__main__":
    x = np.linspace(0, 4, 10)

    fig, axes = plt.subplots(1, 1)
    plt.title("RCParams")
    plt.xlabel("x")
    plt.ylabel("$\delta_1 \\uparrow$")
    for i in range(5):
        y = -np.log(x + 0.001) + i * 3
        axes.plot(x, y, "-o", label=f"Color {i}")
    plt.legend()
    plt.savefig("_rcparams.png")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    img = extract_image_from_fig(fig)
    print("Extracted image shape:", img.shape)

    fig, axes = make_row_of_axes(400, [1, 2, 1], gap_pts=[60, 60], aspect_ratio=0.75)
    for i, ax in enumerate(axes):
        ax.plot(np.random.randn(10))
        ax.set_title(f"Axis {i+1}")
    plt.savefig("_make_row_of_axes.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["A", "B", "C", "D"], [72.4, 81.2, 77.8, 9.5], width=0.65)
    ax.set_ylim(0, 90)
    fig.tight_layout()
    round_bar_tops(ax, bars, radius=10)
    label_bars(ax, bars)
    fig.savefig("_rounded_bars.png")
    plt.show()
