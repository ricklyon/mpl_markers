from . import markers
from . import utils


def onkey_release(event):

    if event.key == "shift":
        event.canvas.figure._marker_hold = True


def onkey_press(event):
    axes = utils.get_event_axes(event)

    toolbar = event.canvas.toolbar
    if event.key == "escape":
        if toolbar.mode == "pan/ ":
            toolbar.pan()
        elif toolbar.mode == "zoom rect":
            toolbar.zoom()
    if axes is None:
        return
    if axes.marker_active is None:
        return
    elif event.key == "shift":
        event.canvas.figure._marker_hold = True
    elif event.key == "left":
        markers.shift_active(axes, -1, call_handler=True)
        markers.draw_active(axes)
    elif event.key == "right":
        markers.shift_active(axes, 1, call_handler=True)
        markers.draw_active(axes)
    elif event.key == "shift+left":
        markers.shift_active(axes, -10, call_handler=True)
        markers.draw_active(axes)
    elif event.key == "shift+right":
        markers.shift_active(axes, 10, call_handler=True)
        markers.draw_active(axes)
    elif event.key == "delete":
        markers.remove(axes)
        markers.draw_all(axes)
    elif event.key == "f5":
        on_draw()


def onmotion(event):
    x = event.xdata
    y = event.ydata
    axes = utils.get_event_axes(event)

    if axes is None or axes.marker_active is None:
        return

    if getattr(axes, "_marker_move", False):
        markers.move_active(x, y, call_handler=True, axes=axes)
        markers.draw_active(axes)


def onclick(event):
    axes = utils.get_event_axes(event)

    if axes is None:
        return

    axes._marker_move = True

    m = utils.get_event_marker(axes, event)
    if m is not None and axes.marker_active != m:
        markers.set_active(axes, m)
        markers.draw_all(axes)


def onrelease(event):
    x = event.xdata
    y = event.ydata
    axes = utils.get_event_axes(event)

    if axes is None:
        return

    axes._marker_move = False

    m = utils.get_event_marker(axes, event)
    active_marker = axes.marker_active

    # create a new marker
    if m is None and (active_marker is None or axes.figure._marker_hold):
        # attempt to create a pcolormesh marker if there are no lines on the plot,
        # if the plot doesn't have a colormesh object, this will return None.
        if not len(axes.lines):
            markers.mesh_marker(x, y, axes=axes)
        # if there are lines on the plot, create a line marker
        else:
            markers.line_marker(x, y, axes=axes)
        markers.draw_all(axes)
    # change the active marker
    elif m is not None:
        markers.set_active(axes, m)
        markers.draw_all(axes)
    # move the active marker
    elif active_marker is not None:
        markers.move_active(x, y, axes=axes, call_handler=True)
        markers.draw_active(axes)
    else:
        return


def on_draw(event):
    markers.init_canvas(event.canvas.figure)

    if not event.canvas.supports_blit:
        # for pdf and svg, we have to use the renderer passed into the event, or else the updates
        # won't apply to the correct figure.
        canvas_draw(event.canvas.figure, event.renderer)
        # make the markers invisible again after the draw
        for axes in event.canvas.figure.axes:
            if hasattr(axes, "markers"):
                [m.set_visible(False) for m in axes.markers]


def canvas_draw(figure, renderer=None):

    figure.canvas.mpl_disconnect(figure._marker_handlers["draw_event"])
    if renderer is not None:
        figure.draw(renderer)
    else:
        figure.canvas.draw()

    figure._marker_handlers["draw_event"] = figure.canvas.mpl_connect(
        "draw_event", on_draw
    )


def init_events(figure):

    event_handlers = dict(
        button_press_event=onclick,
        key_press_event=onkey_press,
        key_release_event=onkey_release,
        motion_notify_event=onmotion,
        button_release_event=onrelease,
        draw_event=on_draw,
    )

    figure._marker_handlers = {}
    figure._marker_hold = False

    for k, v in event_handlers.items():
        figure._marker_handlers[k] = figure.canvas.mpl_connect(k, v)
