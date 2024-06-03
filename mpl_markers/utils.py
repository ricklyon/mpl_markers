from matplotlib.axes import Axes
from matplotlib import ticker
from typing import Tuple, Union, Callable
import numpy as np


def data2display(axes, point):
    return axes.transData.transform(point)


def axes2display(axes, point):
    return axes.transAxes.transform(point)


def display2data(axes, point):
    return axes.transData.inverted().transform(point)


def yformatter(
    xd: float, yd: float, idx: int, axes: Axes, custom: Callable = None
) -> str:
    """
    Returns a formatted string for the y-label marker at the data points xd, yd.
    """
    tick_yformatter = axes.yaxis.get_major_formatter()
    ret = ""

    if isinstance(custom, Callable):
        # try calling with the x, y data points first
        try:
            return custom(xd, yd, idx)
        # call with just the y-data if above failed
        except TypeError:
            return custom(yd)

    # formatting string
    elif isinstance(custom, str):
        return custom.format(yd)

    # use ticker formatter if scalar or fixed formatter
    elif not isinstance(
        tick_yformatter, (ticker.ScalarFormatter, ticker.FixedFormatter)
    ):
        return tick_yformatter(yd)

    # otherwise default to basic format
    else:
        return "{:.3f}".format(yd)


def xformatter(
    xd: float, yd: float, idx: int, axes: Axes, custom: Callable = None
) -> str:
    """
    Returns a formatted string for the x-label marker at the data point xd.
    """
    tick_xformatter = axes.xaxis.get_major_formatter()
    ret = ""

    if isinstance(custom, Callable):
        # try calling with the x, y data points first
        try:
            return custom(xd, yd, idx)
        # call with just the x-data if above failed
        except TypeError:
            return custom(xd)

    # formatting string
    elif isinstance(custom, str):
        return custom.format(xd)

    # use ticker formatter if scalar or fixed formatter
    elif not isinstance(
        tick_xformatter, (ticker.ScalarFormatter, ticker.FixedFormatter)
    ):
        return tick_xformatter(xd)

    # otherwise default to basic format
    else:
        return "{:.3f}".format(xd)


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
