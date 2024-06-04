from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.collections import QuadMesh
import itertools
import json
from pathlib import Path

from . import artists, interactive

__all__ = (
    "line_marker",
    "mesh_marker",
    "axis_marker",
    "set_style",
    "clear",
    "remove",
    "disable_lines",
    "add_handler",
    "draw_active",
    "move_active",
    "draw_all",
    "init_axes",
)

_global_style_file = Path(__file__).parent / "style/default.json"


def line_marker(
    x: float = None,
    y: float = None,
    lines: List[Line2D] = None,
    axes: plt.Axes = None,
    alias_xdata: np.ndarray = None,
    call_handler: bool = False,
    xline: Union[dict, bool] = True,
    yline: Union[dict, bool] = False,
    datadot: Union[dict, bool] = True,
    xlabel: Union[dict, bool] = False,
    ylabel: Union[dict, bool] = True,
    xformatter: Callable = None,
    yformatter: Callable = None,
) -> artists.DataMarker:
    """
    Adds a line marker to cartesian or polar plot.

    Parameters
    ----------
    x: float
        x-axis data value of marker
    y: float (optional)
        y-axis data value
    lines: list (optional)
        list of line2D objects to attach marker to. If not provided, marker will attach to all lines on the axes.
    axes: plt.Axes (optional)
        Axes object to add markers to. Defaults to plt.gca()
    alias_xdata: np.ndarray (optional)
        xdata used for x/y data values in place of the actual x-axis data. Length must match the x-axis data of each line,
        and each line on the marker must have identical lengths.
    call_handler: bool (optional)
        if True, calls the marker handler attached to the axes, if it exists. Defaults to False.
    xline: bool OR dictionary = True
        If True, shows a vertical line at the x value of the marker. If dictionary, parameters are passed
        into Line2D.
    yline: bool OR dictionary = False
        If True, shows a horizontal line at the y value of the marker. If dictionary, parameters are passed
        into Line2D.
    datadot: bool OR dictionary = True
        If True, shows a dot at the data point of the marker. If dictionary, parameters are passed into Line2D
    xlabel: bool OR dictionary = False
        If True, shows a text box of the x value of the marker at the bottom of the axes. If dictionary, parameters are passed
        into axes.text()
    ylabel: bool OR dictionary = True
        If True, shows a text box of the y value of the marker at the data point location. If dictionary, parameters are passed
        into axes.text()
    xformatter: Callable = None
        function that returns a string to be placed in the x-axis label given a x data coordinate
            def xformatter(x: float, idx:int) -> str
    yformatter: Callable = None
        function that returns a string to be placed in the data label given a x and y data coordinate
            def yformatter(x: float, y:float, idx:int) -> str

    Returns:
    --------
    Marker object
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

    # compile artist properties from user provided values or the defaults
    properties = {}
    for k, prop in zip(
        ["xline", "yline", "xlabel", "ylabel", "datadot"],
        [xline, yline, xlabel, ylabel, datadot],
    ):
        # pull default style if True was passed into this property
        if prop is True:
            properties[k] = axes._marker_style[k]
        # override the default with user provided dictionaries
        elif isinstance(prop, dict):
            properties[k] = axes._marker_style[k]
            # allow partial dictionaries
            for n, v in prop.items():
                properties[k][n] = v

    # get the x and y label formatters from the axes if not provided
    if xformatter is None and axes._marker_xformatter:
        xformatter = axes._marker_xformatter
    if yformatter is None and axes._marker_yformatter:
        yformatter = axes._marker_yformatter

    # create marker on the existing data lines
    m = artists.DataMarker(
        axes,
        lines,
        xlabel_formatter=xformatter,
        ylabel_formatter=yformatter,
        alias_xdata=alias_xdata,
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


def mesh_marker(
    x: float = None,
    y: float = None,
    axes: plt.Axes = None,
    xline: Union[dict, bool] = True,
    yline: Union[dict, bool] = True,
    xlabel: Union[dict, bool] = False,
    ylabel: Union[dict, bool] = False,
    zlabel: Union[dict, bool] = True,
    xformatter: Callable = None,
    yformatter: Callable = None,
    zformatter: Callable = None,
    call_handler: bool = False,
) -> artists.MeshMarker:
    """
    Adds new marker on a pcolormesh plot.

    Parameters
    ----------
    x: float
        x-axis value (in data coordinates) of marker
    y: float (optional)
        y-axis value
    axes: plt.Axes (optional)
        Axes object to add markers to. Defaults to plt.gca()
    xline: bool OR dictionary = True
        shows a vertical line at the x value of the marker. If dictionary, parameters are passed
        into Line2D.
    yline: bool OR dictionary = True
        shows a horizontal line at the y value of the marker. If dictionary, parameters are passed
        into Line2D.
    xlabel: bool OR dictionary = False
        shows a text box of the x value of the marker at the bottom of the axes. If dictionary, parameters are passed
        into axes.text()
    ylabel: bool OR dictionary = False
        shows a text box of the y value of the maker along the y axes. If dictionary, parameters are passed
        into axes.text()
    zlabel: bool OR dictionary = True
        shows a text box of the z value of the marker at the data point location. If dictionary, parameters are passed
        into axes.text()
    xformatter: Callable = None
        function that returns a string to be placed in the x-axis label given a x data coordinate
            def xformatter(x: float, idx:int) -> str
    yformatter: Callable = None
        function that returns a string to be placed in the data label given a x and y data coordinate
            def yformatter(x: float, y:float, idx:int) -> str
    zformatter: Callable = None
        function that returns a string to be placed in the z-axis label given a xy data coordinate
            def xformatter(x: float, idx:int) -> str

    Returns:
    --------
    Marker object
    """

    # get current axes if user did not provide one
    if axes is None:
        axes = plt.gca()

    axes = init_axes(axes)

    collection = axes.collections[0] if len(axes.collections) else None

    if not isinstance(collection, QuadMesh):
        return None

    # pull properties from default styles
    properties = {}
    for k, prop in zip(
        ["xline", "yline", "xlabel", "ylabel", "zlabel"],
        [xline, yline, xlabel, ylabel, zlabel],
    ):
        # pull default style if True was passed into this property
        if prop is True:
            properties[k] = axes._marker_style[k]
        # override the default with user provided dictionaries
        elif isinstance(prop, dict):
            properties[k] = axes._marker_style[k]
            # allow partial dictionaries
            for n, v in prop.items():
                properties[k][n] = v

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
    yformatter: Callable = None,
    xformatter: Callable = None,
) -> artists.AxisLabel:
    """
    Adds a marker at the axis edges.

    Parameters
    ----------
    x: float
        x-axis value (in data coordinates) of marker
    y: float (optional)
        y-axis value of marker
    axes: plt.Axes (optional)
        Axes object to add markers to. Defaults to plt.gca()
    ref_marker: Marker (optional):
        reference marker. If provided, the marker will show relative values from the reference.
    xline: bool OR dictionary = True
        shows a vertical line at the x value of the marker. If dictionary, parameters are passed
        into Line2D.
    yline: bool OR dictionary = False
        shows a horizontal line at the y value of the marker. If dictionary, parameters are passed
        into Line2D.
    axisdot: bool OR dictionary = True
        If True, shows a dot at the data point of the marker. If dictionary, parameters are passed into Line2D
    xlabel: bool OR dictionary = False
        shows a text box of the x value of the marker at the bottom of the axes. If dictionary, parameters are passed
        into axes.text()
    ylabel: bool OR dictionary = True
        shows a text box of the y value of the marker at the data point location. If dictionary, parameters are passed
        into axes.text()
    yformatter: Callable = None
        function that returns a string to be placed in the data label given a x and y data coordinate
            def yformatter(x: float, y:float, idx:int) -> str
    xformatter: Callable = None
        function that returns a string to be placed in the x-axis label given a x data coordinate
            def xformatter(x: float, idx:int) -> str

    Returns:
    --------
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
    elif x is None:
        xline = False if x is None else xline
        xlabel = False if x is None else xlabel

    # pull properties from default styles
    properties = {}
    for k, prop in zip(
        ["xline", "yline", "xlabel", "ylabel", "axisdot"],
        [xline, yline, xlabel, ylabel, axisdot],
    ):
        if prop is True or prop is None:
            properties[k] = axes._marker_style[k]
        elif isinstance(prop, dict):
            properties[k] = axes._marker_style[k]
            for n, v in prop.items():
                properties[k][n] = v

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
        **properties,
    )
    m.set_position(x, y)

    # create new marker and append to the axes marker list
    axes.markers.append(m)
    axes.marker_active = m

    return m


def set_style(path: Path):
    """
    Sets the style globally on all future markers.

    Parameters:
    -----------
    path: Path
        path to a .json file that matches the structure of "style/default.json"
    """
    global _global_style_file
    _global_style_file = Path(path)


def clear(axes: plt.Axes = None):
    """
    Removes all markers from axes. To remove a single marker call remove() on Marker object.

    Parameters:
    ----------
    axes: mpl.Axes (optional)
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


def disable_lines(lines: Union[List[Line2D], Line2D], axes: plt.Axes = None):
    """
    Disables markers on each of the provided lines for future markers. Existing markers will not be affected.
    To remove existing markers, use .remove() on the Marker object.

    Parameters:
    ----------
    lines: list OR np.ndarray OR Line2D
        list of Line2D objects that will be ignored by future markers.
    axes: mpl.Axes (optional)
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

    Parameters:
    ---------
    axes: mpl.Axes
        matplotlib axes
    handler: callable:
        def handler(xd: float, yd: float, **kwargs) -> None:
            ...
        The xd and yd parameter are arrays of x-axis/y-axis data values of each line on the active marker.
        kwargs are the same as the optional kwargs passed into add_handler.
    kwargs: dict (Optional)
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
    Moves the active marker to a point along the x-axis.

    Parameters:
    ----------
    x: float
        x-axis value in data coordinates to move the x-marker to.
    y: float (Optional)
        y-axis value in data coordinates to move the x-marker to.
    axes: mpl.Axes
        matplotlib axes object
    disp: bool (Optional)
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

    Parameters:
    ----------
    axes: mpl.Axes
        matplotlib axes object
    direction: int
        incrementally shift active marker along the x-axis left (direction=-1)
        or right (direction=1)
    """

    # do nothing if active marker is not a data marker or has no data labels
    m = axes.marker_active
    if m is None or not isinstance(m, artists.DataMarker):
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
    axes: mpl.Axes
        matplotlib axes object
    blit (bool):
        If True, drawn artists will be blitted onto canvas and the background
        image will be updated. If False, the artists will be drawn but the canvas
        will not be updated. Only used if the canvas supports blitting.
    """
    axes = axes._marker_axes
    [m.update_positions() for m in axes.markers]

    # some backends (like pdf or svg) do not support blitting since they are not interactive backends.
    # all we have to do here is draw the markers on the canvas.
    if not axes.figure.canvas.supports_blit:
        # draw all markers, including the active marker
        [m.draw() for m in axes.markers]
        return

    # raise error if marker_init_canvas has not been called yet
    if axes._all_background is None:
        init_canvas(axes.figure)

    # restore the canvas background with no markers or marker lines drawn on it
    axes.figure.canvas.restore_region(axes._all_background)

    # now draw all the markers on this canvas except the active marker
    [line.axes.draw_artist(line) for line in axes._marker_lines]
    [m.draw() for m in axes.markers if m != axes.marker_active]

    if blit:
        # the active_background has everything draw on it except the active marker.
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
    Configures the figure canvas to support blitting so markers can be updated quickly. This updates the canvas background
    image and should be called whenever the figure canvas is resized or modified.
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
                    ax._legend_background.append(
                        fig.canvas.copy_from_bbox(
                            ax_s.get_legend().get_frame().get_bbox()
                        )
                    )

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
        interactive.canvas_draw(fig, event.renderer)
        for axes in fig._marker_axes:
            [m.set_visible(False) for m in axes.markers]


def init_axes(
    axes: plt.Axes,
    style_path: Path = None,
    xformatter: Callable = None,
    yformatter: Callable = None,
    xline: dict = None,
    yline: dict = None,
    datadot: dict = None,
    axisdot: dict = None,
    xlabel: dict = None,
    ylabel: dict = None,
    handler: Callable[[np.ndarray, np.ndarray, Optional[dict]], None] = None,
) -> plt.Axes:
    """
    Compiles a list of axes lines that will accept marker y-labels and validates the x-axis data of each.

    Parameters
    ----------
    axes: mpl.Axes
        matplotlib axes object
    style_path: Path
        path to a .json file that matches the structure of "style/default.json". Style is
        applied to markers on this axes only.
    xline: bool OR dictionary = True
        shows a vertical line at the x value of the marker. If dictionary, parameters are passed
        into Line2D.
    yline: bool OR dictionary = False
        shows a horizontal line at the y value of the marker. If dictionary, parameters are passed
        into Line2D.
    datadot: bool OR dictionary = True
        shows a dot at the data point of the marker. If dictionary, parameters are passed into Line2D
    axisdot: bool OR dictionary = True
        shows a dot at the location of axis markers. If dictionary, parameters are passed into Line2D
    xlabel: bool OR dictionary = False
        shows a text box of the x value of the marker at the bottom of the axes. If dictionary, parameters are passed
        into axes.text()
    ylabel: bool OR dictionary = True
        shows a text box of the y value of the marker at the data point location. If dictionary, parameters are passed
        into axes.text()
    yformatter: Callable = None
        function that returns a string to be placed in the data label given a x and y data coordinate
            def yformatter(x: float, y:float, idx:int) -> str
    xformatter: Callable = None
        function that returns a string to be placed in the x-axis label given a x data coordinate
            def xformatter(x: float, idx:int) -> str
    handler: callable:
        def handler(xd: float, yd: float, **kwargs) -> None:
            ...

        The xd and yd parameter are arrays of x-axis/y-axis data values of each line on the active marker.
        kwargs are the same as the optional kwargs passed into add_handler.
    """
    axes._marker_axes = axes
    axes._secondary_axes = []

    # load the default style sheet
    if not hasattr(axes, "_marker_style"):
        # read default style
        style_path = _global_style_file if style_path is None else style_path
        with open(style_path) as f:
            axes._marker_style = json.load(f)

    # update the marker styles with the user provided properties
    for k, prop in zip(
        ["xline", "yline", "xlabel", "ylabel", "datadot", "axisdot"],
        [xline, yline, xlabel, ylabel, datadot, axisdot],
    ):
        if prop:
            for n, v in prop.items():
                axes._marker_style[k][n] = v

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
    lines_unfiltered = list(axes.lines) + list(
        itertools.chain.from_iterable([ax.lines for ax in axes._secondary_axes])
    )
    # filter out lines with 2 or fewer data points (straight lines, or single markers)
    axes._marker_lines = [
        ln
        for ln in lines_unfiltered
        if ln not in axes._marker_ignorelines and len(ln.get_xdata()) > 2
    ]

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
