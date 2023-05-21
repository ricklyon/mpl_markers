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

def yformatter(xd: float, yd: float, idx: int, axes: Axes, custom: Callable= None) -> str:
    """
    Returns a formatted string for the y-label marker at the data points xd, yd.
    """
    tick_yformatter = axes.yaxis.get_major_formatter()
    ret = ""

    if custom:
        # try calling with the x, y data points first
        try:
            return custom(xd, yd, idx)
        # call with just the y-data if above failed
        except TypeError:
            return custom(yd)

    # use ticker formatter if scalar or fixed formatter
    if (
        not isinstance(tick_yformatter, (ticker.ScalarFormatter, ticker.FixedFormatter))
    ):
        ret = tick_yformatter(yd)

    # otherwise default to basic format
    if not len(ret):
        return "{:.3f}".format(yd)


def xformatter(xd: float, yd: float, idx: int, axes: Axes, custom: Callable= None) -> str:
    """
    Returns a formatted string for the x-label marker at the data point xd.
    """
    tick_xformatter = axes.xaxis.get_major_formatter()
    ret = ""

    if custom:
        # try calling with the x, y data points first
        try:
            return custom(xd, yd, idx)
        # call with just the x-data if above failed
        except TypeError:
            return custom(xd)

    # use ticker formatter if scalar or fixed formatter
    if (
        not isinstance(tick_xformatter, (ticker.ScalarFormatter, ticker.FixedFormatter))
    ):
        return tick_xformatter(xd)
    
    # otherwise default to basic format
    if not len(ret):
        return "{:.3f}".format(xd)

def get_artist_bbox(artist, padding:Tuple[float, float]=None):
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

def get_event_marker(axes, event):
    for m in axes.markers:
        contains = m.contains(event)
        if contains:
            return m
    return None

def get_event_axes(event):

    axes = event.inaxes
    
    if axes is None:
        return None
        
    tmode = axes.figure.canvas.toolbar.mode
    if tmode != "":
        return None

    try:
        if event.button != 1:
            return None
    except (Exception):
        pass

    axes = event.inaxes

    if hasattr(axes, '_marker_axes'):
        return axes._marker_axes
    else:
        return None