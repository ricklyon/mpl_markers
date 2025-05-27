from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import QuadMesh, PathCollection
import itertools
import json
from pathlib import Path
from copy import deepcopy

from . import artists, interactive, utils

__all__ = (
    "line_marker",
    "mesh_marker",
    "axis_marker",
    "scatter_marker",
    "set_style",
    "set_style_json",
    "clear",
    "remove",
    "disable_lines",
    "add_handler",
    "draw_active",
    "move_active",
    "draw_all",
    "init_axes",
)

_global_style = dict()


def line_marker(
    x: float = None,
    y: float = None,
    idx: int = None,
    lines: List[Line2D] = None,
    axes: plt.Axes = None,
    xline: Union[dict, bool] = None,
    yline: Union[dict, bool] = False,
    datadot: Union[dict, bool] = True,
    xlabel: Union[dict, bool]= False,
    ylabel: Union[dict, bool] = True,
    xformatter: Callable[[float], str] = None,
    yformatter: Callable[[float, float, int], str] = None,
    anchor: str = "center left",
    call_handler: bool = False,
    disp: bool = False,
) -> artists.LineMarker:
    """
    Add a line marker to cartesian or polar plot.

    Parameters
    ----------
    x : float | list, optional
        x-axis data value of marker
    y : float | list, optional
        y-axis data value of marker
    idx : int | list, optional
        index of line data to place marker at. Overrides x and y arguments.
    lines : list, optional
        list of Line2D objects to attach marker to. If not provided, marker will attach to all lines on the axes.
    axes : plt.Axes, optional
        Axes object to add markers to. Default is the current active axes, ``plt.gca()``.
    xline : bool | dict, default: True
        If True, shows a vertical line at the x value of the marker. If dictionary, parameters are passed
        to Line2D.
    yline : bool | dict, default: False
        If True, shows a horizontal line at the y value of the marker. If dictionary, parameters are passed
        to Line2D.
    datadot : bool | dict, default: True
        If True, shows a dot at the data point of the marker. If dictionary, parameters are passed to Line2D.
    xlabel : bool | dict, default: False
        If True, shows a text box of the x value of the marker at the bottom of the axes. If dictionary, parameters are 
        passed into Axes.text()
    ylabel : bool | dict, default: True
        If True, shows a text box of the y value of the marker at the data point location. If dictionary, parameters are 
        passed into Axes.text()
    xformatter : (x: float) -> str, optional
        function that returns a string to be placed in the x-axis label given a x data coordinate. Also accepts
        a string formatter (e.g. "{:.4f}").
    yformatter : (x: float, y: float, idx: int) -> str, optional
        function that returns a string to be placed in the data label given a x, y data coordinate, and the
        index of the line data the marker is located at. Also accepts a string formatter (e.g "{:.4f}").
    anchor : str = None
        anchor location for the y-axis data labels. One of "upper/lower/center left/right/center". Default is
        "center left"
    call_handler : bool, default: False
        if True, calls the marker handler attached to the axes after adding the marker, if it exists. Default is False.

    Returns
    -------
    artists.LineMarker
    """

    # get current axes if user did not provide one
    if axes is None:
        axes = plt.gca()

    axes = init_axes(axes)

    # use lines kwarg if provided, otherwise use all marker lines attached to the axes
    lines = [lines] if isinstance(lines, Line2D) else lines
    lines = axes._marker_lines if lines is None else lines

    if not len(lines):
        return None

    # cast position arguments to arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    idx = np.atleast_1d(idx)

    # broadcast positions (including single float values) to identical shapes. 
    marker_num = max([len(x), len(y), len(idx)])
    x = np.broadcast_to(x, marker_num).copy()
    y = np.broadcast_to(y, marker_num).copy()
    idx = np.broadcast_to(idx, marker_num).copy()

    # check if all lines have monotonic x-axis data vectors. 
    monotonic_xdata = True
    for ln in lines:
        if monotonic_xdata:
            diff = np.diff(ln.get_xdata())
            monotonic_xdata = np.all(diff > 0) or np.all(diff < 0)

    # turn the xline off by default if more than 2 markers are added or any line x-axis data is not monotonic.
    # xlabel is already off by default
    if xline is None:
        xline = True if marker_num < 3 and monotonic_xdata else False

    # compile artist properties from user provided values or the defaults
    properties = utils.compile_properties(
        axes,
        ["xline", "yline", "xlabel", "ylabel", "datadot"],
        [xline, yline, xlabel, ylabel, datadot],
    )

    # get the x and y label formatters from the axes if not provided
    if xformatter is None and axes._marker_xformatter:
        xformatter = axes._marker_xformatter
    if yformatter is None and axes._marker_yformatter:
        yformatter = axes._marker_yformatter

    # create marker(s) on the existing data lines
    m = []
    for i in range(marker_num):
        
        m_i = artists.LineMarker(
            axes,
            lines,
            xlabel_formatter=xformatter,
            ylabel_formatter=yformatter,
            anchor=anchor,
            **properties,
        )

        # append to the axes marker list and set as active marker
        axes.markers.append(m_i)
        axes.marker_active = m_i
        m += [m_i]
        m_i.set_position(x[i], y[i], idx[i], disp=disp)

    # call the axes handler if it exists
    if axes._marker_handler is not None and call_handler:
        func, params = axes._marker_handler
        func(*axes.marker_active.get_data_points(), **params)

    if marker_num == 1:
        return m[0]
    else:
        return m


def mesh_marker(
    x: float = None,
    y: float = None,
    axes: plt.Axes = None,
    xline: Union[dict, bool] = True,
    yline: Union[dict, bool] = True,
    xlabel: Union[dict, bool] = False,
    ylabel: Union[dict, bool] = False,
    zlabel: Union[dict, bool] =  True,
    xformatter: Callable[[float], str] = None,
    yformatter: Callable[[float], str] = None,
    zformatter: Callable[[float], str] = None,
    anchor: str = "center left",
    call_handler: bool = False,
    disp: bool = False
) -> artists.MeshMarker:
    """
    Adds new marker on a pcolormesh plot.

    Parameters
    ----------
    x : float, optional
        x-axis data value of marker
    y : float, optional
        y-axis data value of marker
    axes : plt.Axes, optional
        Axes object to add markers to. Default is the current active axes, ``plt.gca()``.
    xline : bool | dict, default: True
        If True, shows a vertical line at the x value of the marker. If dictionary, parameters are passed
        to Line2D.
    yline : bool | dict, default: True
        If True, shows a horizontal line at the y value of the marker. If dictionary, parameters are passed
        to Line2D.
    xlabel : bool | dict, default: False
        If True, shows a text box of the x value of the marker on the x-axis. If dictionary, parameters are 
        passed into Axes.text()
    ylabel : bool | dict, default: False
        If True, shows a text box of the y value of the marker on the y-axis. If dictionary, parameters are 
        passed into Axes.text()
    zlabel : bool | dict, default: True
        shows a text box of the z value of the marker at the data point location. If dictionary, parameters are passed
        into axes.text()
    xformatter : (x: float) -> str, optional
        function that returns a string to be placed in the x-axis label given a x data coordinate.
        Also accepts a string formatter (e.g. "{:.4f}").
    yformatter : (y: float) -> str, optional
        function that returns a string to be placed in the y-axis label give a y data coordinate.
        Also accepts a string formatter (e.g. "{:.4f}").
    zformatter : (z: float) -> str, optional
        function that returns a string to be placed in the z-axis label given a z data coordinate.
        Also accepts a string formatter (e.g. "{:.4f}").
    anchor : str = None
        anchor location for the data labels. One of "upper/lower/center left/right/center". Default is
        "center left"
    call_handler : bool, default: False
        if True, calls the marker handler attached to the axes after adding the marker, if it exists. Default is False.

    Returns
    -------
    artists.MeshMarker
    """

    # get current axes if user did not provide one
    if axes is None:
        axes = plt.gca()

    axes = init_axes(axes)

    collection = axes.collections[0] if len(axes.collections) else None

    if not isinstance(collection, QuadMesh):
        return None

    # pull properties from default styles
    properties = utils.compile_properties(
        axes,
        ["xline", "yline", "xlabel", "ylabel", "zlabel"],
        [xline, yline, xlabel, ylabel, zlabel],
    )

    # get the x and y label formatters from the axes if not provided
    if xformatter is None and axes._marker_xformatter:
        xformatter = axes._marker_xformatter
    if yformatter is None and axes._marker_yformatter:
        yformatter = axes._marker_yformatter
    if zformatter is None and axes._marker_yformatter:
        zformatter = axes._marker_yformatter

    m = artists.MeshMarker(
        axes,
        collection,
        xlabel_formatter=xformatter,
        ylabel_formatter=yformatter,
        zlabel_formatter=yformatter,
        anchor=anchor,
        **properties,
    )

    # create new marker and append to the axes marker list
    axes.markers.append(m)
    axes.marker_active = m
    m.set_position(x, y, disp)

    # call the axes handler if it exists
    if axes._marker_handler is not None and call_handler:
        func, params = axes._marker_handler
        func(*axes.marker_active.get_data_points(), **params)

    return m


def axis_marker(
    x: float = None,
    y: float = None,
    axes: plt.Axes = None,
    ref_marker: artists.AxisLabel = None,
    xline: Union[dict, bool] = None,
    yline: Union[dict, bool] = None,
    axisdot: Union[dict, bool] = None,
    xlabel: Union[dict, bool] = None,
    ylabel: Union[dict, bool] = None,
    xformatter: Callable[[float], str] = None,
    yformatter: Callable[[float], str] = None,
    placement: str = "lower",
) -> artists.AxisLabel:
    """
    Adds an axis marker that moves freely on the axes, not constrained to data lines. 
    If x and y positions are given, creates a crosshair marker, otherwise creates a vertical line (for y) and 
    a horizontal line (for x). Lines by default have the associated label at the bottom (x) or left (y) side of the
    axes.

    Parameters
    ----------
    x : float, optional
        x-axis value of marker, in data coordinates
    y : float, optional
        y-axis value of marker, in data coordinates
    axes : plt.Axes, optional
        Axes object to add markers to. Default is the current active axes, ``plt.gca()``.
    ref_marker: artists.AxisLabel, optional
        reference marker. If provided, the marker will show relative values from the reference marker.
    xline : bool | dictionary, optional
        shows a vertical line at the x value of the marker. If dictionary, parameters are passed
        into Line2D.
    yline : bool | dictionary, optional
        shows a horizontal line at the y value of the marker. If dictionary, parameters are passed
        into Line2D.
    axisdot: bool | dictionary, optional
        If True, shows a dot where the x and y axis markers meet. If dictionary, parameters are passed into Line2D.
        Ignored if only xline or yline is enabled.
    xlabel : bool | dictionary, optional
        shows a text box of the x value of the marker. If dictionary, parameters are passed into axes.text()
    ylabel : bool | dictionary, optional
        shows a text box of the y value of the marker. If dictionary, parameters are passed into axes.text()
    xformatter : (x: float) -> str, optional
        function that returns a string to be placed in the x-axis label given a x data coordinate.
        Also accepts a string formatter (e.g. "{:.4f}").
    yformatter : (y: float) -> str, optional
        function that returns a string to be placed in the y-axis label give a y data coordinate.
        Also accepts a string formatter (e.g. "{:.4f}").
    placement : {"lower", "upper"}, default: "lower"
        places axis label on the bottom/left side of the axes if "lower", or on the top/right side if "upper".

    Returns
    -------
    Marker object
    """
    # get current axes if user did not provide one
    if axes is None:
        axes = plt.gca()

    axes = init_axes(axes)

    # turn off yline by default if y is not given
    if y is None:
        yline = False if y is None else yline
        ylabel = False if y is None else ylabel
    # turn off xline by default if y is not given
    elif x is None:
        xline = False if x is None else xline
        xlabel = False if x is None else xlabel

    # pull properties from default styles
    properties = utils.compile_properties(
        axes,
        ["xline", "yline", "xlabel", "ylabel", "axisdot"],
        [xline, yline, xlabel, ylabel, axisdot],
    )

    # get the x and y label formatters from the axes if not provided
    if xformatter is None and axes._marker_xformatter:
        xformatter = axes._marker_xformatter
    if yformatter is None and axes._marker_yformatter:
        yformatter = axes._marker_yformatter

    m = artists.AxisLabel(
        axes,
        xlabel_formatter=xformatter,
        ylabel_formatter=yformatter,
        ref_marker=ref_marker,
        placement=placement,
        **properties,
    )

    # create new marker and append to the axes marker list
    axes.markers.append(m)
    axes.marker_active = m

    m.set_position(x, y)

    return m


def scatter_marker(
    x: float = None,
    y: float = None,
    collection: PathCollection = None,
    axes: plt.Axes = None,
    xline: Union[dict, bool] = False,
    yline: Union[dict, bool] = False,
    scatterdot: Union[dict, bool] = True,
    xlabel: Union[dict, bool] = False,
    ylabel: Union[dict, bool] = True,
    xformatter: Callable[[float], str] = None,
    yformatter: Callable[[float, float, int], str] = None,
    anchor: str = "center left",
    call_handler: bool = False,
) -> artists.ScatterMarker:
    """
    Adds a marker to cartesian scatter plot.

    Parameters
    ----------
    x : float
        x-axis data value of marker
    y : float, optional
        y-axis data value
    collection : PathCollection, optional
        PathCollection to attach markers to (returned from plt.scatter()).
        If not provided, uses the first collection found on the axes.
    axes : plt.Axes, optional
        Axes object to add markers to. Default is the current active axes, ``plt.gca()``.
    xline : bool | dictionary, optional
        shows a vertical line at the x value of the marker. If dictionary, parameters are passed into Line2D.
    yline : bool | dictionary, optional
        shows a horizontal line at the y value of the marker. If dictionary, parameters are passed into Line2D.
    scatterdot : bool OR dictionary = True
        If True, shows a dot at the data point of the marker. If dictionary, parameters are passed into Line2D
    xlabel : bool | dictionary, optional
        If True, shows a text box of the x value of the marker at the bottom of the axes. If dictionary, parameters 
        are passed into axes.text()
    ylabel : bool | dictionary, optional
        If True, shows a text box of the y value of the marker at the data point location. If dictionary, parameters 
        are passed into axes.text()
    xformatter : (x: float) -> str, optional
        function that returns a string to be placed in the x-axis label given a x data coordinate. Also accepts
        a string formatter (e.g. "{:.4f}").
    yformatter : (x: float, y: float, idx: int) -> str, optional
        function that returns a string to be placed in the data label given a x, y data coordinate, and the
        index of the data the marker is located at. Also accepts a string formatter (e.g "{:.4f}").
    anchor : str = None
        anchor point for the y-axis data labels. One of "upper/lower/center left/right/center". Default is
        "center left"
    call_handler: bool , default: False
        if True, calls the marker handler attached to the axes, if it exists. Default is False.
    Returns:
    --------
    Marker object
    """

    # get current axes if user did not provide one
    if axes is None:
        axes = plt.gca()

    axes = init_axes(axes)

    # if collection is not provided, search for the first PathCollection found on the axes
    if collection is None:
        c_list = [c for c in axes.collections if isinstance(c, PathCollection)]

        if not len(c_list):
            return None
        else:
            collection = c_list[0]

    properties = utils.compile_properties(
        axes,
        ["xline", "yline", "xlabel", "ylabel", "scatterdot"],
        [xline, yline, xlabel, ylabel, scatterdot],
    )

    # get the x and y label formatters from the axes if not provided
    if yformatter is None and axes._marker_yformatter:
        yformatter = axes._marker_yformatter
    if xformatter is None and axes._marker_xformatter:
        xformatter = axes._marker_xformatter

    # create marker on the existing data lines
    m = artists.ScatterMarker(
        axes,
        collection,
        ylabel_formatter=yformatter,
        anchor=anchor,
        **properties,
    )

    m.set_position(x, y)

    # create new marker and append to the axes marker list
    axes.markers.append(m)
    axes.marker_active = m

    # call the axes handler if it exists
    if axes._marker_handler is not None and call_handler:
        func, params = axes._marker_handler
        func(*axes.marker_active.get_data_points(), **params)

    return m


def set_style_json(path: Path):
    """
    Sets the style globally on all future markers.

    Parameters
    ----------
    path : Path
        path to a .json file that matches the structure of "style/default.json"
    """

    with open(path) as f:
        set_style(**json.load(f))


def set_style(**properties):
    """
    Sets the style globally on all future markers.

    Parameters
    ----------
    ** properties
        any or all key value pairs found in "style/default.json"
    """
    global _global_style

    # start with the default style
    default_style_path = Path(__file__).parent / "style/default.json"
    with open(default_style_path) as f:
        _global_style = json.load(f)

    # overwrite the defaults with the provided properties
    for k in _global_style.keys():
        # each property is a dictionary, leave the default prop intact and only overwrite the provided keys
        if k in properties.keys():
            for pk in properties[k].keys():
                _global_style[k][pk] = deepcopy(properties[k][pk])


def clear(axes: plt.Axes = None):
    """
    Removes all markers from axes. To remove a single marker call ``.remove()`` on Marker object.

    Parameters
    ----------
    axes : plt.Axes, optional
        matplotlib axes object. Defaults to plt.gca()
    """
    # get current axes if user did not provide one
    if axes is None:
        axes = plt.gca()

    # return if markers are not enabled for the axes
    if not hasattr(axes, "_marker_axes"):
        return

    axes = axes._marker_axes

    for m in axes.markers:
        m.remove()

    # clear marker list and active marker
    axes.markers = []
    axes._marker_lines = []
    axes.marker_active = None


def remove(axes: plt.Axes = None, marker: artists.MarkerArtist = None):
    """
    Removes a marker from the axes. If no marker is provided, removes the active marker.
    """
    if marker is None:
        marker = getattr(axes, "marker_active", None)

    # remove the artists associated with this marker
    marker.remove()
    # remove the marker from the marker list
    idx = axes.markers.index(marker)
    axes.markers.pop(idx)
    # set the active marker to the last marker in the list.
    axes.marker_active = axes.markers[-1] if len(axes.markers) else None


def set_active(axes: plt.Axes, marker: artists.MarkerArtist):
    """
    Sets marker as the active marker on the axes.
    """
    if not hasattr(axes, "_marker_axes"):
        return

    if marker not in axes.markers:
        raise ValueError("Marker does not belong to axes.")

    axes.marker_active = marker
    draw_all(axes)


def disable_lines(lines: Union[List, Line2D], axes: plt.Axes = None):
    """
    Disables markers on each of the provided lines for all future markers. Existing markers will not be affected.
    To remove existing markers, use .remove() on the Marker object.

    Parameters:
    ----------
    lines : list | np.ndarray | Line2D
        list of Line2D objects that will be ignored by future markers.
    axes : plt.Axes, optional
        matplotlib axes object. Defaults to plt.gca()
    """
    # cast lines provided as numpy array as a list
    if isinstance(lines, np.ndarray):
        lines = list(lines)
    # cast single line objects as a list
    elif not isinstance(lines, list):
        lines = [lines]

    # get current axes if user did not provide one
    if axes is None:
        axes = plt.gca()

    if hasattr(axes, "marker_axes"):
        axes = axes.marker_axes

    if not hasattr(axes, "_marker_ignorelines"):
        axes._marker_ignorelines = []

    axes._marker_ignorelines += lines


def add_handler(
    axes: plt.Axes,
    handler: Callable[[np.ndarray, np.ndarray, Optional[dict]], None],
    kwargs: dict = None,
):
    """
    Sets a callback function that is called whenever the active marker is moved on the given axes.

    Parameters
    ----------
    axes : mpl.Axes
        matplotlib axes
    handler : (xd: float, yd: float, **kwargs) -> None
        The xd and yd parameter are arrays of x-axis/y-axis data values of each line on the active marker.
        kwargs are the same as the optional kwargs passed into add_handler.
    kwargs : dict, optional
        kwargs that will be passed into the handler every time it's called.
    """
    kwargs = {} if kwargs is None else kwargs

    axes = axes._marker_axes
    # save the function and the parameters as a tuple
    axes._marker_handler = handler, kwargs


def move_active(
    x: float,
    y: float = None,
    call_handler: bool = False,
    axes: plt.Axes = None,
    disp: bool = False,
):
    """
    Moves the active marker to a new point.

    Parameters
    ----------
    x : float
        x-axis value in data coordinates to move the marker to.
    y : float (Optional)
        y-axis value in data coordinates to move the marker to.
    axes : mpl.Axes
        matplotlib axes object
    disp : bool (Optional)
        If True, x and y coordinates are interpreted as display coordinates instead of data coordinates.
    """
    if axes is None:
        axes = plt.gca()

    # do nothing if this axes does not have a marker
    if axes.marker_active is None:
        return

    # move the marker and redraw it
    axes.marker_active.set_position(x=x, y=y, disp=disp)
    draw_active(axes)

    # call the axes handler if it exists
    if axes._marker_handler is not None and call_handler:
        func, params = axes._marker_handler
        func(*axes.marker_active.get_data_points(), **params)


def shift_active(axes: plt.Axes, direction: int, call_handler: bool = False):
    """
    Moves the active marker to a point along the x-axis.

    Parameters
    ----------
    axes : plt.Axes
        matplotlib axes object
    direction : int
        incrementally shift active marker along the x-axis left (direction=-1)
        or right (direction=1)
    """

    # do nothing if active marker is not a data marker or has no data labels
    m = axes.marker_active
    if m is None or not isinstance(m, artists.LineMarker):
        return
    if len(m.data_labels) < 1:
        return

    # use the data index of the first line
    line_label = m.data_labels[0]
    if line_label.idx is None:
        return

    # increment the index of the label, this will only update a single label, but
    # will be overidden by the next call
    line_label.set_position_by_index(line_label.idx + direction)

    # set the position of all the labels based on the current position of the first label
    m.set_position(line_label.xd, line_label.yd)
    draw_active(axes)

    # call the axes handler if it exists
    if axes._marker_handler is not None and call_handler:
        func, params = axes._marker_handler
        func(*axes.marker_active.get_data_points(), **params)


def draw_active(axes: plt.Axes):
    """
    Updates the active marker on the canvas.
    """
    axes = axes._marker_axes

    if not axes._active_background:
        # draw all markers and lines if there is no active background to use
        draw_all(axes)
        return

    if not axes.marker_active:
        # do nothing if there is no active marker
        return

    # erase active marker
    axes.figure.canvas.restore_region(axes._active_background)

    # redraw marker artists
    axes.marker_active.draw()

    # update any markers that reference the active marker
    for m in axes.marker_active.get_dependent_markers():
        m.update_positions()
        m.draw()

    # redraw legends so they are on top of the markers
    for b in axes._legend_background:
        axes.figure.canvas.restore_region(b)

    # apply canvas changes
    axes.figure.canvas.blit(axes.bbox)


def draw_all(axes: plt.Axes, blit: bool = True):
    """
    Updates all markers on the axes canvas, and updates axes active_background image used for blitting. If this
    is called before the canvas has been drawn, it returns silently without updating the canvas.

    Parameters
    ----------
    axes : plot.Axes
        matplotlib axes object
    blit : bool, default: True
        If True, drawn artists will be blitted onto canvas and the background
        image will be updated. If False, the artists will be drawn but the canvas
        will not be updated. Only used if the canvas supports blitting.
    """
    axes = axes._marker_axes
    [m.update_positions() for m in axes.markers]

    # stack any overlaping labels
    utils.stack_ylabels(axes)

    # some backends (like pdf or svg) do not support blitting since they are not interactive backends.
    # all we have to do here is draw the markers on the canvas.
    if not axes.figure.canvas.supports_blit:
        # draw all markers, including the active marker
        [m.draw() for m in axes.markers]
        interactive.canvas_draw(axes.figure)
        return

    # initialize the canvas if not already 
    if axes._all_background is None:
        init_canvas(axes.figure)

    # restore the canvas background with no markers or marker lines drawn on it
    axes.figure.canvas.restore_region(axes._all_background)

    # now draw all the lines and markers on this canvas except the active marker
    [line.axes.draw_artist(line) for line in axes._marker_lines]
    # draw marker objects other than labels first so the labels are on top
    [m.draw_others() for m in axes.markers if m != axes.marker_active]
    [m.draw_labels() for m in axes.markers if m != axes.marker_active]

    if blit:
        # the active_background has everything drawn on it except the active marker.
        # update the canvas first by blitting the other marker artists on it, then save the
        # active background
        axes.figure.canvas.blit(axes.bbox)
        axes._active_background = axes.figure.canvas.copy_from_bbox(axes.bbox)

        # draw the active marker
        if axes.marker_active is not None:
            axes.marker_active.draw()

        # draw the legend over the top of the lines and markers
        for b in axes._legend_background:
            axes.figure.canvas.restore_region(b)

        # update the canvas with the active marker and legend changes.
        axes.figure.canvas.blit(axes.bbox)

    else:
        # if we aren't updating the canvas, draw the active marker on the current canvas
        if axes.marker_active is not None:
            axes.marker_active.draw()

        # bring the legend to the top
        for b in axes._legend_background:
            axes.figure.canvas.restore_region(b)

        # invalidate the active background
        axes._active_background = None


def init_canvas(fig: plt.Figure, event=None):
    """
    Configures the figure canvas to support blitting so markers can be updated quickly. This updates the canvas
    background image and should be called whenever the figure canvas is resized or modified.
    """

    # draw canvas and return if markers have not been initialized
    if not hasattr(fig, "_marker_axes"):
        # fig.canvas.draw()
        return

    if fig.canvas.supports_blit:
        # markers visibility is set to False by default so markers will not be drawn on the canvas here
        for axes in fig._marker_axes:
            [line.set_visible(False) for line in axes._marker_lines]

        interactive.canvas_draw(fig)

        # at this point, we have an updated canvas with no marker artists. Save the bbox for all axes
        # so the markers can be blitted on top of this image later.
        for ax in fig._marker_axes:

            ax._all_background = fig.canvas.copy_from_bbox(ax.bbox)

            # build list of legend bbox so these can be blitted on top of the markers.
            ax._legend_background = []
            # get legends for this axes and any twinx/twiny axes.
            for ax_s in ax._secondary_axes + [ax]:
                if ax_s.get_legend() is not None:
                    ax._legend_background.append(fig.canvas.copy_from_bbox(ax_s.get_legend().get_frame().get_bbox()))

    # draw all markers
    for ax in fig._marker_axes:
        # positions are updated since the canvas was possibly resized or changed
        # in some way since they were placed
        # [m.update_positions() for m in ax.markers]
        [line.set_visible(True) for line in ax._marker_lines]

        # don't use blitting here
        draw_all(ax, blit=False)

    # make the markers invisible again after the draw if blitting is not supported
    if not fig.canvas.supports_blit:
        # for pdf and svg, we have to use the renderer passed into the event, or else the updates
        # won't apply to the correct figure.
        renderer = getattr(event, "renderer", None)
        interactive.canvas_draw(fig, renderer)
        for axes in fig._marker_axes:
            [m.set_visible(False) for m in axes.markers]


def init_axes(
    axes: plt.Axes,
    xformatter: Callable = None,
    yformatter: Callable = None,
    handler: Callable[[np.ndarray, np.ndarray, Optional[dict]], None] = None,
    **properties,
) -> plt.Axes:
    """
    Initializes the axes to accept marker objects

    Parameters
    ----------
    axes : mpl.Axes
        matplotlib axes object
    xformatter : (x: float) -> str, optional
        function that returns a string to be placed in the x-axis label given a x data coordinate.
        Also accepts a string formatter (e.g. "{:.4f}").
    yformatter : (y: float) -> str, optional
        function that returns a string to be placed in the y-axis label give a y data coordinate.
        Also accepts a string formatter (e.g. "{:.4f}").
    handler : (xd: float, yd: float, **kwargs) -> None
        The xd and yd parameter are arrays of x-axis/y-axis data values of each line on the active marker.
        kwargs are the same as the optional kwargs passed into add_handler.
    ** properties
        any or all key value pairs found in style/default.json

    """
    axes._marker_axes = axes
    axes._secondary_axes = []

    # update the marker styles with the user provided properties
    if not len(_global_style):
        set_style()

    if not hasattr(axes, "_marker_style"):
        axes._marker_style = deepcopy(_global_style)

    # overwrite the defaults with the provided properties
    for k in _global_style.keys():
        # each property is a dictionary, leave the default prop intact and only overwrite the provided keys
        if k in properties.keys():
            for pk in properties[k].keys():
                axes._marker_style[k][pk] = deepcopy(properties[k][pk])

    if not hasattr(axes, "_marker_ignorelines"):
        axes._marker_ignorelines = []

    if yformatter is not None or not hasattr(axes, "_marker_yformatter"):
        axes._marker_yformatter = yformatter
    if yformatter is not None or not hasattr(axes, "_marker_xformatter"):
        axes._marker_xformatter = xformatter

    # check if there are other axes that share the same canvas (twinx or twiny axes)
    for ax_p in axes.figure.axes:
        if (ax_p is not axes) and (ax_p.bbox.bounds == axes.bbox.bounds):

            if hasattr(ax_p, "markers"):
                # found a shared axes that has already been initialized-- defer all marker events to this axes.
                axes._marker_axes = ax_p
                # keep track of ignored lines at the primary axes level
                if hasattr(ax_p, "_marker_ignorelines"):
                    axes._marker_ignorelines += ax_p._marker_ignorelines
                # skip initialization for secondary axes
                continue
            else:
                # we found a shared axes, but markers haven't been initialized yet. Treat ax_p as a secondary axes
                axes._secondary_axes.append(ax_p)
                ax_p._marker_axes = axes

        # initialize axes member variables if they don't exist yet
        if not hasattr(axes, "markers"):
            axes.markers = []
            axes.set_zorder(1)

            axes.marker_active = None
            axes._active_background = None
            axes._all_background = None
            axes._marker_handler = None
            axes._legend_background = []

    # do not modify the axes if called on a secondary axes, initialize primary axes instead
    axes = axes._marker_axes

    # compile list of all lines in this axes and any axes that share the same canvas
    lines_unfiltered = list(axes.lines) + list(itertools.chain.from_iterable([ax.lines for ax in axes._secondary_axes]))

    # filter out lines with 2 or fewer data points (straight lines, or single markers), and lines with no valid data
    axes._marker_lines = []
    for ln in lines_unfiltered:
        if ln not in axes._marker_ignorelines and len(ln.get_xdata()) > 2 and np.any(np.isfinite(ln.get_xdata())):
            axes._marker_lines += [ln]

    # add handle to top-level axes to figure
    if not hasattr(axes.figure, "_marker_axes"):
        axes.figure._marker_axes = [axes]
    elif axes not in axes.figure._marker_axes:
        axes.figure._marker_axes.append(axes)

    # create marker handler if in kwargs
    if handler is not None:
        add_handler(axes, handler)

    # add interactive event handlers
    interactive.init_events(axes.figure)

    init_canvas(axes.figure)

    return axes._marker_axes
