# mpl-markers

Interactive data markers for line plots in matplotlib

## Installation

```bash
pip install mpl-markers
```

## Usage

```python
import mpl_markers as mplm
```

### Line Markers
Add a marker attached to plotted data lines:
```python
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rc('font', size=7)

fig, ax = plt.subplots(1,1)
x1 = np.linspace(-2*np.pi, 2*np.pi, 1000)

ax.plot(x1, np.sin(x1)*np.cos(x1)**2)

mplm.data_marker(x=0)
```
The marker can be dragged to any location along the data line, or moved incrementally with the left/right arrow keys.

![example1](https://raw.githubusercontent.com/ricklyon/mpl_markers/main/docs/img/example1.gif)

### Axis Markers
Add an axis marker that moves freely on the canvas:
```python
ax.xaxis.set_major_formatter(lambda x, pos: '{:.2f}$\pi$'.format(x/np.pi))
mplm.axis_marker(x=0, y=-0.2)
```

![example2](https://raw.githubusercontent.com/ricklyon/mpl_markers/main/docs/img/example2.gif)

`set_major_formatter` will set the formatting for the axes ticks and the marker label. To only 
format the label, use the following,
```python
mplm.axis_marker(x=0, y=-0.2, xformatter="{:.2f}$\pi$", yformatter="{:.2f}$\pi$")
```

### Meshgrid Markers
Data markers can also be added to `pcolormesh` plots. The marker label shows the value of the color-mapped z data.

```python
xy = np.linspace(-1, 1, 100)
x, y = np.meshgrid(xy, xy)
z = np.sin(2*x)**2 + np.cos(3*y)**2

fig, ax = plt.subplots(1, 1)
m = ax.pcolormesh(x, y, z, vmin=0, vmax=2)
plt.colorbar(m)

# add a data marker at a single x/y point on the plot. x/y is in data coordinates.
mplm.data_marker(x=0.75, y=0.25)
```
![example3](https://raw.githubusercontent.com/ricklyon/mpl_markers/main/docs/img/example3.gif)

## Styling
The marker style is controlled by the `mpl_markers/style/default.json` file:

```json
{
    "xline": {
        "linewidth": 0.6,
        "color": "k",
        "linestyle": "dashed"
    },
    "yline": {
        "linewidth": 0.6,
        "color": "k",
        "linestyle": "dashed"
    },
    "xlabel": {
        "fontsize": 8,
        "color": "black",
        "bbox": {
            "boxstyle": "square",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 1,
            "linewidth": 1.5
        }
    },
    "ylabel": {
        "fontsize": 8,
        "bbox": {
            "boxstyle": "square",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 1,
            "linewidth": 1.5
        }
    },
    "xydot": {
        "markersize": 10,
        "marker": "."
    },
    "xymark": {
        "markersize": 10,
        "marker": ".",
        "markerfacecolor":"white", 
        "markeredgecolor":"k"
    }
}

```
To use custom styles, pass in a dictionary of artist settings when creating the marker that matches the keys in this file.
For example, this line will change the color of the dotted vertical line to red.

```python
mplm.data_marker(x=0, xline=dict(color='red', linestyle="dashed", alpha=0.5))
```

To turn on/off any of the artists, pass in `True/False` for the artist key,
```python
mplm.data_marker(x=0, xlabel=True, xline=False)
```

## License

mpl-markers is licensed under the MIT License.