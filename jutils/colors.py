import numpy as np


class Colors:
    def __init__(self):
        self.lmu = "#00883a"
        self.cvgroup = "#131285"
        self.spezi = ['#49277d', '#e3027e', '#e10612', '#ec6605', '#fbb901']
        self.fivemap = ['#FFD700', '#ff7f0e', '#d62728', '#9467bd', '#1f77b4']
        self.pyplot = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def __repr__(self):
        return_str = ""
        for color in self.__dict__.keys():
            dtype = type(self.__dict__[color])
            if dtype == str:
                return_str += f"{color:<12}: {self.__dict__[color]}\n"
            elif dtype == list:
                length = len(self.__dict__[color])
                return_str += f"{color + f' ({length})':<12}: {self.__dict__[color]}\n"
        return return_str
    

JCOLORS = Colors()


""" Functions """


def hex_to_rgb(hex):
    assert isinstance(hex, str), "Hex color must be a string"
    if hex[0] == '#': hex = hex.lstrip('#')
    assert len(hex) == 6, "Hex color must be 6 characters long"
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))


def interpolate_color_list(n, clist=None):
    if clist is None:
        clist = JCOLORS.spezi
    if isinstance(clist[0], str):       # convert from hex to rgb
        clist = [hex_to_rgb(color) for color in clist]
    if isinstance(clist[0][0], int):    # convert from rgb to float
        clist = [(r/255, g/255, b/255) for r, g, b in clist]

    c = len(clist)
    if n < c: return clist[:n]
    
    total_new_colors = n - c
    num_segments = c - 1
    colors_per_segment = total_new_colors // num_segments
    remainder = total_new_colors % num_segments
    
    result = []
    for i in range(c - 1):
        start_color = np.array(clist[i])
        end_color = np.array(clist[i + 1])
        result.append(clist[i])
        segment_colors = colors_per_segment + (1 if i < remainder else 0)
        for t in range(1, segment_colors + 1):
            interp_color = start_color + (end_color - start_color) * (t / (segment_colors + 1))
            result.append(tuple(interp_color))
    result.append(clist[-1])
    return result


def interpolate_colors(n, color1=(0.60156, 0, 0.99218), color2=(0.86328, 0.47656, 0.31250)):
    red_difference = color2[0]-color1[0]
    green_difference = color2[1]-color1[1]
    blue_difference = color2[2]-color1[2]

    red_delta = red_difference/n
    green_delta = green_difference/n
    blue_delta = blue_difference/n

    _colors = [
        (color1[0] + (red_delta * i), color1[1] + (green_delta * i), color1[2] + (blue_delta * i))
        for i in range(n)
    ]
    return _colors


def visualize_color_list(clist, ax=None):
    if ax is None: fig, ax = plt.subplots()
    for i, c in enumerate(clist):
        ax.plot([0, 1], [i, i], color=c, linewidth=10)
    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # _Colors
    print(JCOLORS)

    # interpolate and visualize colors
    interpolated_colors = interpolate_color_list(10, JCOLORS.spezi)
    fix, ax = plt.subplots()
    visualize_color_list(interpolated_colors, ax)
    plt.savefig("_interpolate_color_list.png")

    # interpolate_colors
    colors = interpolate_colors(10)
    fig, ax = plt.subplots()
    visualize_color_list(colors, ax)
    plt.savefig("_interpolate_colors.png")
