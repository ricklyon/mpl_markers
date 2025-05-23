from matplotlib.axes import Axes
from matplotlib import ticker
from typing import Tuple, Union, Callable, List
import numpy as np
from copy import deepcopy
import itertools


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
    Returns a formatted string for the marker at the data points xd, yd.
    """

    if custom is not None:
        if isinstance(custom, Callable):
            if mode == "x":
                return custom(xd)
            else:
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


def deconflict_ylabels(axes):
    """
    
    """
    ax_pad = axes.figure.dpi / 10
    # use half the normal label pad since each label has a pad and it will be doubled
    # when labels are stacked on top of each other.
    label_pad = axes.figure.dpi / 20

    pad = np.array([label_pad, label_pad])

    # get all adaptive label artists from all markers
    labels = []
    for m in axes.markers:
        labels += [obj for obj in m.adaptive_artists if not obj._hidden]

    # make axes smaller with negative padding
    ax_bbox = get_artist_bbox(axes, (-ax_pad, -ax_pad))
    # determine upper and lower edge of the axes space
    upper_y = ax_bbox[1, 1]
    lower_y = ax_bbox[0, 1]

    # make groups of labels with overlapping x bounds, ignore y bounds for now
    groups = []
    free_labels = list(labels)

    for i, a in enumerate(labels):
        for b in free_labels:

            if a == b:
                continue

            a_x = get_artist_bbox(a, pad)[:, 0]
            b_x = get_artist_bbox(b, pad)[:, 0]

            overlap_x = False
            # do boxes overlap in x?
            # a is left of b, but right side of a overlaps
            if (a_x[0] < b_x[0]) and (a_x[1] > b_x[0]): 
                overlap_x = True
            # a is right of b, but left side of a overlaps
            if (a_x[0] > b_x[0]) and (a_x[0] < b_x[1]): 
                overlap_x = True

            if overlap_x:
                free_labels.remove(b)

                if a in free_labels:
                    free_labels.remove(a)
                    # create new group since both a and b are free
                    groups += [(a, b)]
                else:
                    # get the group index where a belongs and add b to the same group
                    group_idx = [j for j, g in enumerate(groups) if a in g][0]
                    groups[group_idx] += (b,)

    # create groups with a single label for all left over free labels
    groups += [[lbl] for lbl in free_labels]
            
    for g in groups:

        # sort each label in ascending y order (using the bottom left corner y value)
        sorted_g = sorted(g, key=lambda x: get_artist_bbox(x, pad)[0, 1])
        # build an array of all label y bounds
        g_y = np.array([get_artist_bbox(a, pad)[:, 1] for a in sorted_g])

        # start in the middle, and push labels up
        g_mid = len(g) // 2
        for i in range(len(g[g_mid:])):
            if g_mid + i >= len(g) - 1:
                break
            
            # get bbox for both labels, a is below b
            a_y = g_y[g_mid + i]
            b_y = g_y[g_mid + i + 1]

            overlap = a_y[1] - b_y[0]
            # if overlap is positive, shift b box up to remove overlap
            if overlap > 0:
                g_y[g_mid + i + 1] += overlap
        
        # start again in the middle, this time pushing labels down
        for i in range(len(g[:g_mid + 1])):
            if (g_mid - i - 1) < 0:
                break
            
            # get bbox for both labels, a is above b
            a_y = g_y[g_mid - i]
            b_y = g_y[g_mid - i - 1]

            overlap = b_y[1] - a_y[0]
            # if overlap is positive, shift b down up to remove overlap
            if overlap > 0:
                g_y[g_mid - i - 1] -= overlap
    
        # start from the top and push labels down if there is overlap with the top axes
        overlap = g_y[-1][0] - upper_y

        for i in range(len(g)):

            if overlap > 0:
                g_y[-1 - i] -= overlap
            # force label within upper axes limit, even if it causes overlap above it
            if g_y[-1 - i][0] < lower_y:
                g_y[i] += (lower_y - g_y[-1 - i][0])

            if i >= len(g) - 1:
                break   
            # get overlap with next label, a is above b
            a_y = g_y[-1 - i]
            b_y = g_y[-2 - i]
            overlap = b_y[1] - a_y[0]

        # start from the bottom and push labels up if there is overlap with the bottom axes
        overlap = lower_y - g_y[0][1]

        for i in range(len(g)):

            if overlap > 0:
                g_y[i] += overlap
            # force label within upper axes limit, even if it causes overlap below it
            if g_y[i][1] > upper_y:
                g_y[i] -= (g_y[i][1] - upper_y)
            
            if i >= len(g) - 1:
                break   
            # get overlap with next label, b is above a
            a_y = g_y[i]
            b_y = g_y[i + 1]
            overlap = a_y[1] - b_y[0]

        # set the positions for all labels in the group
        for i, lbl in enumerate(sorted_g):
            lbl_x = get_artist_bbox(lbl)[0, 0]
            lbl.set_position((lbl_x, g_y[i, 0] + pad[1]), disp=True, anchor="lower left")
