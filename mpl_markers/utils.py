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

    bbox definition:   
    [[x0, y0]  
     [x1, y1]]

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


def get_bbox_overlap(a_bbox, b_bbox, y_order=None):
    """
    Returns the amount of overlap between a and b labels, in display coordinates.
    Positive values indicate the amount of overlap, negative values are margin.
    """
    a_x, a_y = a_bbox.T
    b_x, b_y = b_bbox.T

    # do boxes overlap in x?
    # a is left of b
    if (a_x[0] < b_x[0]): 
        overlap_x = a_x[1] - b_x[0]
    # a is right of b
    else:
        overlap_x = b_x[1] - a_x[0]

    # do boxes overlap in y?
    # a is below b
    if (y_order is None and (a_y[0] < b_y[0])) or y_order == "ab": 
        overlap_y = a_y[1] - b_y[0]
    # a is above b
    else:
        overlap_y = b_y[1] - a_y[0]

    return overlap_x, overlap_y


def get_label_overlap(a, b, pad):
    """
    Returns the amount of overlap between a and b labels, in display coordinates.
    Positive values indicate the amount of overlap, negative values are margin.
    """
    a_bbox = get_artist_bbox(a, pad)
    b_bbox = get_artist_bbox(b, pad)

    return get_bbox_overlap(a_bbox, b_bbox)


def push_bbox(a, b, y_order="ab"):
    """
    Returns a new bbox for "a" that is shifted up or down to avoid "b". 

    Parameters
    ----------
    a : np.ndarray
        bbox of label to push
    b : np.ndarray
        bbox of label to avoid
    y_order : {"ab", "ba"}
        If a should remain below b, use "ab". If a should remain above b, use "ba
    """
    overlap_x, overlap_y = get_bbox_overlap(a, b, y_order=y_order)

    if overlap_y > 0 and overlap_x > 0:
        # a is above b, shift a up to avoid b
        if y_order == "ba":
            a[:, 1] += overlap_y

        # a is below b, shift a down to avoid b
        else:
            a[:, 1] -= overlap_y

    return a
        

def group_labels_by_xoverlap(labels: list, pad: tuple) -> dict:
    """
    Separates labels into columns that do not overlap in the x direction with other groups. 
    Labels within a group can be vertically moved without overlapping with labels from other groups. 
    """

    groups = []

    for i, a in enumerate(labels):
        
        a_x = get_artist_bbox(a, pad)[:, 0]
        ovl_groups = []
        # check if a overlaps with any existing groups
        for i, g in enumerate(groups):

            b_x = g["bounds"]
            # a is left of b OR a is right of b
            overlap_x = a_x[1] - b_x[0] if (a_x[0] < b_x[0]) else b_x[1] - a_x[0]
            
            # if a overlaps in x direction, add to group
            if overlap_x > 0:
                # add a to group
                g["labels"] += [a]
                g["bounds"] = min([a_x[0], b_x[0]]), max([a_x[1], b_x[1]])
                ovl_groups += [i]

        # labels can fall into multiple groups, merge groups with shared labels
        if len(ovl_groups) > 1:
            g0 = groups[ovl_groups[0]]
            for g_i in ovl_groups[1:]:
                g = groups[g_i]
                g0["bounds"] = min([g0["bounds"][0], g["bounds"][0]]), max([g0["bounds"][1], g["bounds"][1]])
                g0["labels"] += [lbl for lbl in g["labels"] if lbl is not a]

            # remove old groups that were merged into first group
            [groups.pop(g) for g in sorted(ovl_groups[1:], reverse=True)]

        # create a new group if a does not overlap with any existing ones
        if len(ovl_groups) < 1:
            groups += [dict(labels=[a], bounds=get_artist_bbox(a, pad)[:, 0])]

    return groups


def stack_ylabels(axes, markers: list=None):
    """
    Vertically stack overlapping labels associated with markers. If markers is not given, stacks labels for
    all markers found on the axes.
    """

    if markers is None:
        markers = axes.markers

    # get all label artists from all markers
    labels = []
    for m in markers:
        labels += [obj for obj in m._label_artists if not obj._hidden]

    if not len(labels):
        return 
    
    # apply padding from first label to all labels.
    padding = labels[0]._padding * (axes.figure.dpi / 100)
    pad = np.array([padding, padding])

    groups = group_labels_by_xoverlap(labels, pad)
    # drop the x-bounds, get list of labels in each group
    groups = [g["labels"] for g in groups]

    # make axes smaller with negative padding
    ax_bbox = get_artist_bbox(axes, (-padding / 2, -padding / 2))

    for g in groups:

        if len(g) < 2:
            continue

        # sort each label in ascending y order. Use the last xy display coordinates from a set_position call
        # that was not from deconflict labels. This keeps the labels in absolute ascending y order.
        sorted_g = sorted(g, key=lambda x: x._persistent_y)

        # build an array of all label bboxes
        g_bbox = np.array([get_artist_bbox(a, pad) for a in sorted_g])

        # start in the middle, and push labels up
        g_mid = len(g) // 2
        for i in range(len(g) - g_mid):

            a = g_mid + i

            # push "a" label up if any "b" label below it overlaps 
            for j in range(i + 1):
                
                b = a - (j + 1)
                # a is above b
                push_bbox(g_bbox[a], g_bbox[b], y_order="ba")
                    
        # start again in the middle, this time pushing labels down
        for i in range(g_mid):
            a = g_mid - i - 1

            # push "a" label down if any "b" label above it overlaps 
            for j in range(i + 1):

                b = a + (j + 1)
                # a is below b
                push_bbox(g_bbox[a], g_bbox[b], y_order="ab")
    
        # start from the top and push labels down if there is overlap with the top axes
        for i in range(len(g)):
            a = len(g) -i - 1

            # move labels that are above the axes bounds down into the axes
            ax_overlap_top =  g_bbox[a][1, 1] - ax_bbox[1, 1]
            if ax_overlap_top > 0:
                g_bbox[a, :, 1] -= ax_overlap_top

            # push "a" label down if any "b" label above it overlaps 
            for j in range(i + 1):

                b = a + (j + 1)
                if b > (len(g) - 1):
                    break

                # a is below b
                push_bbox(g_bbox[a], g_bbox[b], y_order="ab")

        # start from the bottom and push labels up if there is overlap with the bottom axes
        for i in range(len(g)):
            a = i

            # move labels that are below the axes bounds up into the axes
            ax_overlap_btm =  ax_bbox[0, 1] - g_bbox[a][0, 1]
            if ax_overlap_btm > 0:
                g_bbox[a, :, 1] += ax_overlap_btm

            # push "a" label up if any "b" label below it overlaps 
            for j in range(i + 1):

                b = a - (j + 1)
                if b < 0:
                    break

                # a is above b
                push_bbox(g_bbox[a], g_bbox[b], y_order="ba")

        # set the positions for all labels in the group
        for i, lbl in enumerate(sorted_g):
            lbl.set_position(g_bbox[i][0] + pad, disp=True, anchor="lower left", persist=False)
