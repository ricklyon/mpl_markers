# mpl-markers

Interactive data markers for matplotlib.

## Installation

```bash
pip install mpl-markers
```

## Usage

```python
import mpl_markers as mplm
```

### Line Markers
Add a marker attached to matplotlib data lines:
```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1)
x1 = np.linspace(-np.pi, np.pi, 1000)

ax.plot(x1, np.sin(x1)*np.cos(x1)**2)
# create line marker at x=0.
mplm.line_marker(x=0)
```
In interactive matplotlib backends (i.e. Qt5Agg), the marker can be dragged to any location along the data line, or moved incrementally with the left/right arrow keys. Interactive markers are not supported for inline figures 
generated in Jupyter Notebooks.

![example1](https://raw.githubusercontent.com/ricklyon/mpl_markers/main/docs/img/example1.gif)

Additional markers can be added by using the Shift+Left Mouse button. The active marker can be removed from the plot by pressing the Delete key.

### Axis Markers
Axis markers move freely on the canvas and are not attached to data lines. Axis markers can
reference other markers to create a delta marker.
```python
fig, ax = plt.subplots(1, 1)
y1 = np.random.normal(6, 3, size=10)

ax.bar(np.arange(10), y1)
ax.margins(x=0.2)

# create horizontal axis marker
m1 = mplm.axis_marker(y=np.min(y1), yformatter="{:.2f}%")

# create second marker that is referenced from the first marker m1
mplm.axis_marker(y=np.max(y1), ref_marker=m1, yformatter="{:.2f}%")
```
![example2](https://raw.githubusercontent.com/ricklyon/mpl_markers/main/docs/img/example2.png)

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
mplm.mesh_marker(x=0.75, y=0)
```
![example3](https://raw.githubusercontent.com/ricklyon/mpl_markers/main/docs/img/example3.gif)

### Scatter Plot Marker

Markers attached to scatter plots can be added with `scatter_marker`.

```python
data_x = np.random.normal(-1, 1, 100)
data_y = np.random.normal(-1, 1, 100)

fig, ax = plt.subplots(1, 1)
s = ax.scatter(data_x, data_y, color="b")

# place the marker on the point closest to x=0, y=0.5.
# if there is only one scatter plot on the axes, the collection argument can be dropped.
mplm.scatter_marker(
    0, 0.5, collection=s, yformatter=lambda x, y, pos: f"x={x:.2f}\ny={y:.2f}", anchor="upper left"
)
```
![example5](https://raw.githubusercontent.com/ricklyon/mpl_markers/main/docs/img/example5.gif)


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
    "zlabel": {
        "fontsize": 8,
        "bbox": {
            "boxstyle": "square",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 1,
            "linewidth": 1.5
        }
    },
    "datadot": {
        "markersize": 10,
        "marker": "."
    },
    "axisdot": {
        "markersize": 10,
        "marker": ".",
        "markerfacecolor":"white", 
        "markeredgecolor":"k"
    },
    "scatterdot": {
        "markersize": 10,
        "marker": ".",
        "markeredgewidth": 1,
        "markeredgecolor": "white"
    }
}

```
To use custom styles, pass in a dictionary of artist settings to the `set_style` method. Keys that do not match those found in `default.json` are ignored. Settings are applied globally to all future markers.

```python
mplm.set_style(
    ylabel=dict(fontfamily="monospace", bbox=dict(linewidth=0, facecolor="none"))
)
```

To limit the style scope to a specific axes, artist settings can also be passed into `init_axes` as kwargs. Settings can also be passed to individual markers when they are created, and allows for mixed styles on the same axes. 

```python
# Inherits global settings from the last set_style call.
mplm.line_marker(
    x=np.pi/4,
    ylabel=dict(fontsize=11),             
    xline=False # turn off the xline artist
)
```
![example3](https://raw.githubusercontent.com/ricklyon/mpl_markers/main/docs/img/example4.png)

Custom .json files are supported with the `mplm.set_style_json` method. This will set the style on all future markers and must have the same keys as the default.json file:

```python
mplm.set_style_json("user_style.json")
```

## License

mpl-markers is licensed under the MIT License.