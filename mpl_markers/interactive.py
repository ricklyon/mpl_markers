from . import markers
from .artists import MarkerArtist
import matplotlib.pyplot as plt


def get_event_marker(axes: plt.Axes, event) -> MarkerArtist:
    """
    Returns the marker that was clicked on in from the axes event.
    """
    for m in axes.markers:
        contains = m.contains(event)
        if contains:
            return m
    return None


def get_event_axes(event) -> plt.Axes:
    """
    Returns the axes that triggered the button click event.
    Acts as an event mask, only returns axes if triggered by a left button click,
    and if no toolbar functions are active.
    """
    axes = event.inaxes

    if axes is None:
        return None

    # return None if any toolbar function is active
    toolbar_mode = axes.figure.canvas.toolbar.mode
    if toolbar_mode != "":
        return None

    # only return axes if the event was triggered by a left button click.
    try:
        if event.button != 1:
            return None
    except Exception:
        pass

    # return the top level marker axes (only different if there is a twinx/twiny axes)
    if hasattr(axes, "_marker_axes"):
        return axes._marker_axes
    else:
        return None


def onkey_release(event):
    """
    Clears flag when shift is released.
    """
    if event.key == "shift":
        event.canvas.figure._marker_hold = False


def onkey_press(event):
    """
    Event handler for all key press events.
    """
    axes = get_event_axes(event)

    # if escape was pressed, exit from the zoom or pan toolbar functions
    if event.key == "escape":
        toolbar = event.canvas.toolbar
        if toolbar.mode == "pan/zoom" or toolbar.mode == "pan/ ":
            toolbar.pan()
        elif toolbar.mode == "zoom rect":
            toolbar.zoom()

    # refresh canvas (redraw)
    elif event.key == "f5":
        on_draw(event)

    # exit if event is not within an axes region or there is no active marker
    if axes is None or axes.marker_active is None:
        return

    # set flag if shift is pressed
    if event.key == "shift":
        event.canvas.figure._marker_hold = True
    # increment active marker to the left
    elif event.key == "left":
        markers.shift_active(axes, -1, call_handler=True)
        markers.draw_active(axes)
    # increment active marker to the right
    elif event.key == "right":
        markers.shift_active(axes, 1, call_handler=True)
        markers.draw_active(axes)
    # coarse increment active marker to the left
    elif event.key == "shift+left":
        markers.shift_active(axes, -10, call_handler=True)
        markers.draw_active(axes)
    # coarse increment active marker to the right
    elif event.key == "shift+right":
        markers.shift_active(axes, 10, call_handler=True)
        markers.draw_active(axes)
    # delete active marker
    elif event.key == "delete":
        markers.remove(axes)
        markers.draw_all(axes)


def onmotion(event):
    """
    Drags the active marker if the left mouse button is held.
    """
    x = event.xdata
    y = event.ydata
    axes = get_event_axes(event)

    if axes is None or axes.marker_active is None:
        return

    if getattr(axes, "_marker_move", False):
        markers.move_active(x, y, call_handler=True, axes=axes)
        markers.draw_active(axes)


def onclick(event):
    """
    Event handler for left mouse button clicks. Activates the marker that was clicked on.
    """
    axes = get_event_axes(event)

    if axes is None:
        return

    # set flag that mouse button is held down
    axes._marker_move = True

    # set the active marker
    m = get_event_marker(axes, event)
    if m is not None and axes.marker_active != m:
        markers.set_active(axes, m)


def onrelease(event):
    """
    Event handler for left mouse button releases. Moves the active marker to the location where button
    was released.
    """
    x = event.xdata
    y = event.ydata
    axes = get_event_axes(event)

    if axes is None:
        return
    # clear flag since mouse button is no longer held down
    axes._marker_move = False

    m = get_event_marker(axes, event)
    active_marker = axes.marker_active

    # create a new marker
    if (m is None and active_marker is None) or axes.figure._marker_hold:
        # attempt to create a pcolormesh marker if there are no lines on the plot,
        # if the plot doesn't have a colormesh object, this will return None.
        if not len(axes.lines):
            markers.mesh_marker(x, y, axes=axes)
        # if there are lines on the plot, create a line marker
        else:
            markers.line_marker(x, y, axes=axes)
        markers.draw_all(axes)
    # move the active marker
    elif active_marker is not None:
        markers.move_active(x, y, axes=axes, call_handler=True)
        markers.draw_active(axes)
    else:
        return


def on_draw(event):
    """
    Triggered whenever the canvas is drawn (i.e. resize events).
    """
    markers.init_canvas(event.canvas.figure, event)


def canvas_draw(figure: plt.Figure, renderer=None):
    """
    Draws the figure canvas without triggering the interactive draw event.
    """
    figure.canvas.mpl_disconnect(figure._marker_handlers["draw_event"])

    if renderer is not None:
        figure.draw(renderer)
    else:
        figure.canvas.draw()

    figure._marker_handlers["draw_event"] = figure.canvas.mpl_connect(
        "draw_event", on_draw
    )


def init_events(figure: plt.Figure):
    """
    Registers the events with the figure and initializes flags.
    """

    event_handlers = dict(
        button_press_event=onclick,
        key_press_event=onkey_press,
        motion_notify_event=onmotion,
        button_release_event=onrelease,
        draw_event=on_draw,
    )

    figure._marker_handlers = {}
    figure._marker_hold = False

    for k, v in event_handlers.items():
        figure._marker_handlers[k] = figure.canvas.mpl_connect(k, v)
