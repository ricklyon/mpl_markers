from matplotlib.axes import Axes
from matplotlib import ticker
from typing import Tuple, Union, Callable, List
import numpy as np
from copy import deepcopy


def data2display(axes, point):
    return axes.transData.transform(point)


def axes2display(axes, point):
    return axes.transAxes.transform(point)


def display2data(axes, point):
    return axes.transData.inverted().transform(point)


def label_formatter(
    axes: Axes,
    xd: float,
    yd: float,
    idx: int = None,
    custom: Union[Callable, str] = None,
    mode: str = "x",
    precision: int = "{:.3f}",
) -> str:
    """
    Returns a formatted string for the y-label marker at the data points xd, yd.
    """

    if custom is not None:
        if isinstance(custom, Callable):
            # try calling with the x, y data points first
            try:
                return custom(xd, yd, idx)
            # call with just the x-data or y-data if above failed
            except TypeError:
                return custom(xd if mode == "x" else yd)

        # custom formatting string
        else:
            return custom.format(xd if mode == "x" else yd)

    # attempt to use tick formatter
    axis = axes.xaxis if mode == "x" else axes.yaxis
    tick_formatter = axis.get_major_formatter()
    tick_value = tick_formatter(xd if mode == "x" else yd)

    if isinstance(tick_formatter, ticker.LogFormatter):
        # log tick formatter returns an empty string if not near a tick marker,
        # implement it manually
        lbl = "{:.2e}".format(xd if mode == "x" else yd)
        e_idx = lbl.index("e")
        val, multiplier = lbl[:e_idx], int(lbl[e_idx + 1 :])
        return "{}$\\times 10^{{{}}}$".format(val, multiplier)

    # use ticker formatter if not a scalar
    elif not isinstance(tick_formatter, ticker.ScalarFormatter) and tick_value:
        return tick_value

    # otherwise default to basic format
    else:
        return precision.format(xd if mode == "x" else yd)


def get_artist_bbox(artist, padding: Tuple[float, float] = None):
    """
    Returns the bounding box for the artist in display coordinates, optionally with padding applied

    bbox definition: [x0, y0]
                        [x1, y1]
    (x0, y0) is lower left point of artist, (x1, y1) is upper right corner
    """
    if hasattr(artist.axes.figure.canvas, "get_renderer"):
        renderer = artist.axes.figure.canvas.get_renderer()
    else:
        renderer = None

    vis = artist.get_visible()
    artist.set_visible(True)
    bbox = np.array(artist.get_window_extent(renderer).get_points())
    artist.set_visible(vis)

    if np.any(padding):
        pad_ = np.atleast_1d(padding)
        # bottom left corner needs padding subtracting so bbox is extended left and lower,
        # upper right needs padding added
        bbox += np.array([-pad_, pad_])

    return bbox


def compile_properties(axes, keys: List[str], props: List[dict]):
    """
    Compile artist properties from user provided values or the defaults on the axes
    """
    properties = {}

    for k, p in zip(keys, props):
        # pull default style if True was passed into this property
        if p is True or p is None:
            properties[k] = deepcopy(axes._marker_style[k])
        # override the default with user provided dictionaries
        elif isinstance(p, dict):
            properties[k] = deepcopy(axes._marker_style[k])
            # allow partial dictionaries
            for n, v in p.items():
                properties[k][n] = deepcopy(v)
        # don't add to properties if False was passed in for this property

    return properties
