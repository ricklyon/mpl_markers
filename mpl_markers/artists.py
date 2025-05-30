import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.collections import QuadMesh, PathCollection
from copy import deepcopy as dcopy
from typing import Tuple, Union, Callable
import itertools
import matplotlib
import matplotlib.pyplot as plt


from . import utils


class AbstractArtist(object):
    def __init__(self, axes=None):

        self.axes = axes

        self.set_visible(False)
        self._hidden = False

    def draw_artist(self, renderer=None):
        """Draw each artist associated with marker."""

        canvas = self.axes.figure.canvas

        if self._hidden:
            return

        elif renderer:
            # super calls the next method in the MRO, which should be Text or Line2D
            super().draw(renderer)

        elif canvas.supports_blit:
            self.set_visible(True)
            self.axes.draw_artist(self)
            self.set_visible(False)

        # pdf and svg don't support draw_artist
        else:
            self.set_visible(True)

    def contains(self, event):
        if self._hidden:
            return None

        self.set_visible(True)
        ret = super().contains(event)
        self.set_visible(False)
        return ret

    def set_hidden(self, state):
        self._hidden = state


class MarkerLabel(AbstractArtist, matplotlib.text.Text):

    def __init__(self, axes=None, persistent_y=None, padding=5, offset=0, **kwargs):
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

        # this calls the __init__ method for the class in the MRO after AbstractArtist, so mpl.Text
        # let the label be visible outside the axes for polar plots with clip_on=False
        super(AbstractArtist, self).__init__(0, 0, "0", transform=None, clip_on=False, **kwargs)
        # call the AbstractArtist init method
        super(MarkerLabel, self).__init__(axes)

        # get axes from kwargs and add to axes
        axes.add_artist(self)
        self._is_fixed = bool(persistent_y is not None)
        self._persistent_y = persistent_y if persistent_y is not None else 0

        self._padding = padding
        self._offset_padding = offset

    def set_position(
        self,
        point: Tuple[float, float],
        text: str = None,
        anchor: str = "upper left",
        disp: bool = False,
        offset: Tuple[float, float] = None,
        persist: bool = True,
    ):
        """
        Set label position and text

        archor kwarg specifies where the placement point is on the label:
        'upper/lower/center left/right/center'
        """
        offset = (0, 0) if offset is None else offset
        # set text first so label is sized correctly
        if text is not None:
            self.set_text(text)

        # convert to display coordinates
        point = utils.data2display(self.axes, point) if not disp else point

        # get the label size
        xy0, xy1 = utils.get_artist_bbox(self)
        xlen, ylen = xy1 - xy0

        # set_position places the label's lower left corner at the point. If the user specifies the placement point
        # at 'upper center' for example, the placement needs to be offset by x=+xlen and y=+ylen/2 to put the lower 
        # center of the label at the 'point' argument location.
        # matplotlib display coordinate system is (0,0) at lower left and (width, height) at the upper right
        # https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html

        # first word in loc is vertical position, second is horizontal
        loc_v, loc_h = anchor.split()
        # move the label up by the offset if the anchor is at the bottom, down by the offset if anchor is top,
        # same thing for left/right placement.
        v_offset = {
            "upper": -ylen - offset[1],
            "center": -(ylen / 2),
            "lower": offset[1],
        }[loc_v]
        h_offset = {
            "left": offset[0],
            "center": -(xlen / 2),
            "right": -xlen - offset[0],
        }[loc_h]

        # apply correction to find the position of the lower left corner
        b_left, b_lower = np.array(point) + np.array([h_offset, v_offset])

        # build bbox positions for a label placed at the upper left point
        l_bbox = np.array([[b_left, b_lower], [b_left + xlen, b_lower + ylen]])

        # force label within axes bounds
        ax_pad = np.array([self._padding * (self.axes.figure.dpi / 100)] * 2)

        # make axes smaller with negative padding
        ax_bbox = utils.get_artist_bbox(self.axes, -ax_pad)

        # arrays to subtract label bbox from axes bbox
        ax_sub = np.array([[1, 1], [-1, -1]])
        l_sub = np.array([[-1, -1], [1, 1]])
        # distance that each point extrudes past the axes limits. If the label is fully within axes, the
        # value will be negative and clip will force it to 0.
        bbox_extrude = np.clip((l_bbox * l_sub) + (ax_bbox * ax_sub), 0, None)

        extrude = (bbox_extrude > 0).astype("int")

        if np.any(offset):
            # flip the anchor point to the other side of the label if it extends past the axes limit.
            # these dictionaries determine how much the label should be shifted to flip the anchor point.
            v_ax_shift = {
                "upper": ylen + (2 * offset[1]),
                "center": (ylen / 2) + offset[1],
                "lower": ylen + (2 * offset[1]),
            }[loc_v]

            h_ax_shift = {
                "left": xlen + (2 * offset[0]),
                "center": (xlen / 2) + offset[0],
                "right": xlen + (2 * offset[0]),
            }[loc_h]

            # apply the shift to the label bbox
            l_bbox_shift = np.array([[h_ax_shift, v_ax_shift], [-h_ax_shift, -v_ax_shift]])
            l_bbox += np.sum(l_bbox_shift * extrude, axis=0)

        else:
            # if there is no offset provided, ignore the anchor point and force the label to be in bounds
            l_bbox += bbox_extrude[0]
            l_bbox -= bbox_extrude[1]

        # use lower left bbox point for placement
        # AbstractArtist should not have a set_position method, so this will use set_position from mpl.Text
        super().set_position(l_bbox[0])

        if persist and not self._is_fixed:
            self._persistent_y = l_bbox[0][1]
        

class MarkerLine(AbstractArtist, Line2D):

    def __init__(self, axes=None, **kwargs):
        super(AbstractArtist, self).__init__([0], [0], **kwargs)
        super(MarkerLine, self).__init__(axes)

        axes.add_line(self)


class MarkerArtist(object):
    """
    Abstract class for a group of artists that are associated with a single marker object.
    """

    def __init__(self, axes, artists, dependent_markers=[]):

        self.axes = axes
        self._artists = artists
        self._label_artists = [a for a in artists if isinstance(a, MarkerLabel)]
        self._dependent_markers = dependent_markers
        self._hidden = False

    def remove(self):
        [obj.remove() for obj in self._artists if obj]

    def contains(self, event):

        for obj in self._artists:
            if not obj:
                continue
            cont = obj.contains(event)
            # check if not None and first value is True
            if cont is not None and cont[0]:
                return True
        return False

    def draw(self):
        """Draw each artist associated with marker."""
        [obj.draw_artist() for obj in self._artists if obj]

    def draw_labels(self):
        [obj.draw_artist() for obj in self._label_artists if obj]

    def draw_others(self):
        [obj.draw_artist() for obj in self._artists if obj and obj not in self._label_artists]

    def set_visible(self, state):
        [obj.set_visible(state) for obj in self._artists if obj]

    def set_hidden(self, state):
        [obj.set_hidden(state) for obj in self._artists if obj]
        self._hidden = state

    def get_dependent_markers(self):
        return self._dependent_markers

    def add_dependent_marker(self, marker):
        self._dependent_markers.append(marker)

class LineLabel(MarkerArtist):
    """
    Text label attached to a data point of a Line2D object.
    """

    def __init__(
        self,
        axes: Axes,
        data_line: Line2D,
        datadot: dict = None,
        ylabel: dict = None,
        yline: dict = None,
        ylabel_formatter: Callable = None,
        anchor: str = "center left",
    ):
        """
        Parameters:
        ----------
        axes: Axes
            axes object that marker will be drawn on.
        data_line: Line2D
            line object to attach marker to. Line does not need to belong to axes if twinx or twiny axes are used.
        dot: bool or dictionary
            If true, a default dot will be drawn. If a dictionary, the parameters will be
            passed into the Line2D constructor. False for no marker dot.
        """

        self.ylabel = None
        self.datadot = None
        self.yline = None
        self.ylabel_formatter = ylabel_formatter
        self.data_line = data_line

        self._idx = None
        self._xd = None
        self._yd = None
        self._anchor = anchor

        if ylabel:
            # change the edge color of the text box to the line color
            ylabel = dcopy(ylabel)
            ylabel["bbox"]["edgecolor"] = data_line.get_color()

            # initalize text box as the label
            self.ylabel = MarkerLabel(axes=axes, verticalalignment="bottom", **ylabel)

        if datadot:
            # initalize marker dot at data point
            self.datadot = MarkerLine(axes=data_line.axes, color=data_line.get_color(), **datadot)

        if yline:
            self.yline = MarkerLine(axes=data_line.axes, **yline)

        artists = [self.yline, self.datadot, self.ylabel]

        super().__init__(axes, artists)

    @property
    def idx(self):
        return self._idx

    @property
    def xd(self):
        return self._xd

    @property
    def yd(self):
        return self._yd

    def set_position_by_index(self, idx):

        xdata = self.data_line.get_xdata()
        ydata = self.data_line.get_ydata()

        if idx >= len(xdata):
            idx = len(xdata) - 1

        if idx < 0:
            idx = 0

        self._idx = idx
        self._xd = np.real(xdata[idx])
        self._yd = np.real(ydata[idx])

        # get label position in display coordinates. Use the line axes instead of the class axes since
        # the line may belong to another twinx/y axes and have different scaling.
        xl, yl = utils.data2display(self.data_line.axes, (self._xd, self._yd))

        # hide marker if data is not finite or is NaN
        if np.any(~np.isfinite([self._yd, self._xd, xl, yl])):
            self.set_hidden(True)
            return
        else:
            self.set_hidden(False)

        # set label to left side of axes if yline is active
        xl = self.axes.get_xlim()[0] if self.yline else xl

        if self.ylabel:

            # pad values in display coordinates (pixels)
            label_pad = self.ylabel._offset_padding * (self.axes.figure.dpi / 100)

            # set the x position to the data point location plus a small pad (in display coordinates)
            txt = utils.label_formatter(self.axes, self._xd, self._yd, self._idx, self.ylabel_formatter, mode="y")

            self.ylabel.set_position(
                (xl, yl),
                txt,
                anchor=self._anchor,
                disp=True,
                offset=(label_pad, label_pad) if not self.yline else None,
            )

        if self.yline:
            self.yline.set_data(self.axes.get_xlim(), [self._yd] * 2)

        if self.datadot:
            self.datadot.set_data([self._xd], [self._yd])

    def set_position(self, x: float = None, y: float = None, idx: int = None, disp: bool = False, mode: str = "xy"):
        """
        Sets the position of the label to the nearest data point to x/y.
        """

        # ignore positional arguments if index is given
        if idx is not None:
            self.set_position_by_index(idx)
            return

        if self.axes.name == "polar" and mode == "x":
            # use data coordinates to place marker in "x" mode (places markers by angle)
            if disp:
                x, _ = utils.display2data(self.axes, (x, y))

            x = (x + np.pi) % (2 * np.pi) - np.pi
            # get line data in data coordinates
            xdata, ydata = self.data_line.get_data()
        # convert line data to display coordinates to match the x/y coords
        elif disp:
            ln_xy = utils.data2display(self.axes, self.data_line.get_xydata())
            xdata, ydata = ln_xy.T
        # if x/y are in data coordinates, get the line data in data coordinates
        else:
            xdata, ydata = self.data_line.get_data()

        # compute distances from x/y to all data points
        if x is not None and y is not None and mode == "xy": 
            dist = np.sqrt((x - xdata) ** 2 + (y - ydata) ** 2)
        elif x is not None:
            dist = np.abs(x - xdata)
        elif y is not None:
            dist = np.abs(y - ydata)
        else:
            raise ValueError(f"Insufficient positional arguments for marker.")

        # set position to the data point with the smallest distance
        self.set_position_by_index(np.nanargmin(dist))


class AxisLabel(MarkerArtist):
    """
    Places markers on the x and y axis. Placement is not constrained to data points.
    """

    def __init__(
        self,
        axes: Axes,
        xlabel: dict = None,
        ylabel: dict = None,
        xline: dict = None,
        yline: dict = None,
        axisdot: dict = None,
        xlabel_formatter: Callable = None,
        ylabel_formatter: Callable = None,
        ref_marker: MarkerArtist = None,
        placement: str = "lower",
    ):

        self.xlabel = None
        self.ylabel = None
        self.xline = None
        self.yline = None
        self.axisdot = None
        self.xlabel_formatter = xlabel_formatter
        self.ylabel_formatter = ylabel_formatter
        self.ref_marker = ref_marker
        self.placement = placement
        self._polar = axes.name == "polar"

        if (ylabel or yline) and self._polar:
            raise NotImplementedError("y-axis markers not supported for polar plots")
        
        persistent_y = -2**16 if placement == "lower" else 2**16

        if xline:
            self.xline = MarkerLine(axes=axes, **xline)

        if xlabel:
            self.xlabel = MarkerLabel(axes=axes, verticalalignment="bottom", persistent_y=persistent_y, **xlabel)

        if yline:
            self.yline = MarkerLine(axes=axes, **yline)

        if ylabel:
            self.ylabel = MarkerLabel(axes=axes, verticalalignment="bottom", **ylabel)

        if axisdot and xline and yline:
            # initalize marker dot at data point
            self.axisdot = MarkerLine(axes=axes, **axisdot)

        self._xd = 0
        self._yd = 0

        artists = [self.xline, self.yline, self.axisdot, self.xlabel, self.ylabel]
        super().__init__(axes, artists)

        if ref_marker:
            ref_marker.add_dependent_marker(self)

    def set_position(
        self,
        x: float = None,
        y: float = None,
        disp: bool = False,
    ):

        self._position_args = (x, y, disp)

        if disp:
            x, y = utils.display2data(self.axes, (x, y))

        # set x-axis marker
        if x is not None:
            # force within axes bounds

            if self._polar:
                # wrap xdata between -180 and 180 if axes is polar
                x = (x + np.pi) % (2 * np.pi) - np.pi
            else:
                x = np.clip(x, *self.axes.get_xlim())

            self._xd = x

            if self.xlabel:
                # use reference data if available
                if self.ref_marker:
                    lbl_xd = self._xd - self.ref_marker._xd
                    lbl_sgn = r"$(\Delta)+$" if lbl_xd > 0 else r"$(\Delta)-$"
                    txt = utils.label_formatter(
                        self.axes, np.abs(lbl_xd), self._yd, custom=self.xlabel_formatter, mode="x",
                    )
                    lbl = lbl_sgn + txt
                else:
                    lbl = utils.label_formatter(
                        self.axes, self._xd, self._yd, custom=self.xlabel_formatter, mode="x",
                    )

                ylim_i = dict(lower=0, upper=1)[self.placement]
                xl, yl = utils.data2display(self.axes, (x, self.axes.get_ylim()[ylim_i]))

                self.xlabel.set_position(
                    (xl, yl), lbl, anchor="lower center", disp=True,
                )

            if self.xline:
                self.xline.set_data([x] * 2, self.axes.get_ylim())

        # set y-axis marker
        if y is not None:
            # force within axes bounds
            y = np.clip(y, *self.axes.get_ylim())
            self._yd = y

            xlim_i = dict(lower=0, upper=1)[self.placement]
            xl, yl = utils.data2display(self.axes, (self.axes.get_xlim()[xlim_i], y))

            if self.ylabel:
                # use reference data if available
                if self.ref_marker:
                    lbl_yd = self._yd - self.ref_marker._yd
                    lbl_sgn = r"$(\Delta)+$" if lbl_yd > 0 else r"$(\Delta)-$"

                    txt = utils.label_formatter(
                        self.axes, self._xd, np.abs(lbl_yd), custom=self.ylabel_formatter, mode="y",
                    )
                    lbl = lbl_sgn + txt
                else:
                    lbl = utils.label_formatter(
                        self.axes, self._xd, self._yd, custom=self.ylabel_formatter, mode="y",
                    )

                self.ylabel.set_position(
                    (xl, yl), lbl, anchor="center left", disp=True,
                )

            if self.yline:
                self.yline.set_data(self.axes.get_xlim(), [y] * 2)

        if x is not None and y is not None and self.axisdot:
            self.axisdot.set_data([x], [y])
        elif x is not None and self.axisdot:
            self.axisdot.set_data([x], self.axes.get_ylim()[0])
        elif y is not None and self.axisdot:
            self.axisdot.set_data(self.axes.get_xlim()[0], [y])

    def update_positions(self):
        self.set_position(*self._position_args)
        

class MeshLabel(MarkerArtist):
    """
    Places a marker label on a pcolormesh plot.
    """

    def __init__(
        self,
        axes: Axes,
        quadmesh: QuadMesh,
        zlabel: dict = None,
        zlabel_formatter: Callable = None,
        anchor: str = "center left",
    ):
        """
        Parameters:
        ----------
        axes: Axes
            axes object that marker will be drawn on.
        data_line: Line2D
            line object to attach marker to. Line does not need to belong to axes if twinx or twiny axes are used.
        dot: bool or dictionary
            If true, a default dot will be drawn. If a dictionary, the parameters will be
            passed into the Line2D constructor. False for no marker dot.
        """

        self.zlabel = None
        self.datadot = None
        self.zlabel_formatter = zlabel_formatter
        self.quadmesh = quadmesh
        self.coords = (quadmesh.get_coordinates().data[:-1, :-1] + quadmesh.get_coordinates().data[1:, 1:]) / 2

        self._yidx = None
        self._xidx = None
        self._xd = None
        self._yd = None
        self._anchor = anchor

        if zlabel:
            # initalize text box as the label
            self.zlabel = MarkerLabel(axes=axes, verticalalignment="bottom", **zlabel)

        # initalize marker dot at data point
        self.datadot = MarkerLine(
            axes=axes,
            markeredgewidth=1,
            markeredgecolor="k",
            markerfacecolor="w",
            markersize=10,
            marker=".",
        )
        self.xydot_outer = MarkerLine(
            axes=axes,
            markeredgewidth=1,
            markeredgecolor="k",
            markerfacecolor="w",
            markersize=20,
            marker=".",
        )

        artists = [self.xydot_outer, self.datadot, self.zlabel]

        super().__init__(axes, artists)
        # set to arbitrary position to intialize member variables.
        self.set_position_by_index(xidx=0, yidx=0)

    @property
    def xd(self):
        return self._xd

    @property
    def yd(self):
        return self._yd

    def set_position_by_index(self, xidx, yidx):
        # coordinates is a NxMx2 array where NxM is the shape of the meshgrid.
        # the last dimension holds the xy data coordinates of each meshgrid cell.
        xlen, ylen = self.quadmesh.get_array().shape

        if xidx >= xlen:
            xidx = xlen - 1

        if xidx < 0:
            xidx = 0

        if yidx >= ylen:
            yidx = ylen - 1

        if yidx < 0:
            yidx = 0

        self._xidx, self._yidx = xidx, yidx
        # index the x and y coordinates
        self._xd, self._yd = self.coords[xidx, yidx]

        # get the data value at the xy coordinates
        self._zd = self.quadmesh.get_array()[xidx, yidx]

        # hide marker if data is not finite or NaN
        if not np.isfinite(self._zd):
            self.set_hidden(True)
            return
        else:
            self.set_hidden(False)

        # get label position in display coordinates. Use the line axes instead of the class axes since
        # the line may belong to another twinx/y axes and have different scaling.
        xl, yl = utils.data2display(self.axes, (self._xd, self._yd))

        if self.zlabel:
            # pad values in display coordinates (pixels)
            label_pad = self.zlabel._padding * (self.axes.figure.dpi / 100)

            # set the x position to the data point location plus a small pad (in display coordinates)
            txt = utils.label_formatter(
                self.axes, self._xd, self._zd, self._xidx, self.zlabel_formatter, mode="y",
            )

            self.zlabel.set_position(
                (xl, yl), txt, anchor=self._anchor, disp=True, offset=(label_pad, label_pad),
            )

        if self.datadot:
            self.datadot.set_data([self._xd], [self._yd])
            self.xydot_outer.set_data([self._xd], [self._yd])
            # set the color of the dot
            z_norm = self.quadmesh.norm(self._zd)
            plt.setp(self.datadot, markerfacecolor=self.quadmesh.cmap(z_norm))
            self.datadot.set_color(self.quadmesh.cmap(z_norm))

    def set_position(self, x: float, y: float, disp: bool = False):
        """
        Returns the index of the line data for the point closest to the x,y data values, and the distance in data 
        coordinates
        """
        if disp:
            x, y = utils.display2data(self.axes, (x, y))

        dist = np.argmin(np.sum(np.abs(self.coords - np.array([x, y])[None, None]), axis=-1))

        self._xidx, self._yidx = np.unravel_index(dist, self.coords.shape[:2])

        # set position to the data point with the smallest error
        self.set_position_by_index(self._xidx, self._yidx)


class LineMarker(MarkerArtist):
    """
    Marker attached to multiple lines.
    """

    def __init__(
        self,
        axes: Axes,
        lines: list[Line2D],
        xlabel: dict = None,
        ylabel: dict = None,
        datadot: dict = None,
        xline: dict = None,
        yline: dict = None,
        xlabel_formatter: Callable = None,
        ylabel_formatter: Callable = None,
        anchor: str = "center left",
    ):
        """
        Parameters:
        -----------
        axes : plt.Axes
            Axes object
        lines : list
            list of Line2D objects to attach marker to. 
        xlabel : dict, optional
            Axes.text() properties for x-axis label.
        ylabel : dict, optional
            Axes.text() properties for y-axis label.
        datadot : dict, optional
            Line2D properties for marker dot.
        xline : bool | dict, optional
            Line2D properties for vertical line at marker location
        yline : dict, optional
            Line2D properties for a horizontal line at the marker locations.
        xlabel_formatter : (x: float) -> str, optional
            function that returns a string to be placed in the x-axis label given a x data coordinate. Also accepts
            a string formatter (e.g. "{:.4f}").
        ylabel_formatter : (x: float, y: float, idx: int) -> str, optional
            function that returns a string to be placed in the data label given a x, y data coordinate, and the
            index of the line data the marker is located at. Also accepts a string formatter (e.g "{:.4f}").
        anchor : str = None
            anchor location for the y-axis data labels. One of "upper/lower/center left/right/center". Default is
            "center left"
        """
        self.data_labels = []
        artists = []
        self.xaxis_label = None
        self.axes = axes
        self._anchor = anchor
        # x position of xlabel
        self._xlbl = None
        self._line_idx = None

        self.lines = lines

        # check that ylines have associated labels or dots
        if yline and not (ylabel or datadot):
            raise ValueError("y-axis lines require labels or marker dots.")

        # create single x-axes label shared by all the data markers
        if xline or xlabel:
            self.xaxis_label = AxisLabel(axes, xlabel, False, xline, False, False, xlabel_formatter, None)

        # data labels for each line
        if ylabel or datadot or yline:
            # turn off ylabel on data markers if yline is present. The axes label will be used as the data label.
            self.data_labels = [
                LineLabel(axes, ln, datadot, ylabel, yline, ylabel_formatter, anchor=anchor) for ln in self.lines
            ]

        # build list of all artists in the marker

        # line and dot artists first
        if self.xaxis_label:
            artists += self.xaxis_label._artists[:2]
        artists += list(itertools.chain.from_iterable([lbl._artists[:-1] for lbl in self.data_labels]))

        # then label artists on top
        artists += list(itertools.chain.from_iterable([lbl._artists[-1:] for lbl in self.data_labels]))
        if self.xaxis_label:
            artists += self.xaxis_label._artists[2:]

        self._has_xlabel = self.xaxis_label and self.xaxis_label.xlabel

        super().__init__(axes, artists)

    def set_position(self, x: float, y: float = None, idx: int = None, disp: bool = False):
        """
        Set the position of all line markers to the nearest data point to x/y. 

        Parameters:
        -----------
        x : float
            x position of marker, in data coordinates unless disp=True.
        y : float
            y position of marker, in data coordinates unless disp=True. If marker has a xline object, the y coordinate
            is ignored.
        idx : float, optional
            index of line data to place marker at. All line data must be the same length if using this parameter.
            Overrides x, y parameters.
        disp : bool, default = False
            if True, x and y are interpreted in display coordinates.
        
        """
        # check for invalid position combinations
        if x is None and self.xaxis_label:
            raise ValueError("x-position is required when a xlabel or xline is attached to a marker.")

        # initialize location coordinates for all markers, and the line data index the marker is located at
        xd_yd = np.zeros((2, len(self.lines)))
        line_idx = np.zeros(len(self.lines), dtype=np.int64)

        # ignore the y position if an xlabel is present, use only the x position
        if self.xaxis_label:
            mode = "x"
        else:
            mode = "xy"

        # place ylabels
        for i, lbl in enumerate(self.data_labels):
            lbl.set_position(x, y, idx, disp, mode=mode)
            # save the actual position of the label in data coordinates
            xd_yd[:, i] = lbl.xd, lbl.yd
            line_idx[i] = lbl._idx
            

        self._xd, self._yd = xd_yd
        self._line_idx = line_idx
        # place the x-axis label. Since there are multiple lines attached to the same x-line, put this at the line
        # data point closest to x.
        if self.xaxis_label:

            if idx is not None:
                nearest_lbl_idx = idx

            else:
                if not disp and self.axes.name == "polar":
                    # wrap xdata between -180 and 180 if axes is polar
                    x = (x + np.pi) % (2 * np.pi) - np.pi

                # convert display coordinates to data
                if disp:
                    x, y = utils.display2data(self.axes, (x, y))
                
                if mode == "x":
                    # find the label with the closest x data point to the target position, ignore y
                    nearest_lbl_idx = np.nanargmin(np.abs(self._xd - x))
                else:
                    # find label with closest x and y data point to target
                    nearest_lbl_idx = np.nanargmin(np.sqrt((self._xd - x) ** 2 + (self._yd - y) ** 2))

            self._xlbl = self._xd[nearest_lbl_idx]
            self.xaxis_label.set_position(self._xlbl)

        # stack the labels if they overlap for this label only. All other markers might overlap with this one.
        utils.stack_ylabels(self.axes, [self])

    def get_data_points(self):
        return self._xd, self._yd

    def update_positions(self):

        # place ylabels
        for i, lbl in enumerate(self.data_labels):
            if not lbl._hidden:
                lbl.set_position_by_index(int(self._line_idx[i]))

        # place xlabel
        if self.xaxis_label:
            self.xaxis_label.set_position(self._xlbl)

class MeshMarker(MarkerArtist):
    """
    Places a marker on a pcolormesh axes.
    """

    def __init__(
        self,
        axes: Axes,
        quadmesh: QuadMesh,
        xlabel: dict = None,
        ylabel: dict = None,
        zlabel: dict = None,
        xline: dict = None,
        yline: dict = None,
        xlabel_formatter: Callable = None,
        ylabel_formatter: Callable = None,
        zlabel_formatter: Callable = None,
        anchor: str = None,
    ):
        """
        Parameters:
        -----------


        """
        self.data_label = None
        artists = []
        self.axis_label = None
        self.quadmesh = quadmesh
        self.data_label = None
        self.axes = axes

        # create single x/y-axes label
        if xline or xlabel or yline or ylabel:
            self.axis_label = AxisLabel(
                axes, xlabel, ylabel, xline, yline, False, xlabel_formatter, ylabel_formatter,
            )

        # label for the z data
        self.data_label = MeshLabel(axes, quadmesh, zlabel, zlabel_formatter, anchor=anchor)

        # build list of all artists in the marker
        # line and dot artists first
        if self.axis_label:
            artists += self.axis_label._artists

        artists += self.data_label._artists

        self.set_position(0, 0)

        super().__init__(axes, artists)

    def set_position(self, x: float, y: float, disp: bool = False):
        """
        Parameters:
        -----------

        """

        # save the arguments to update the marker position when the axes bbox changes later
        self._position_args = (x, y, disp)

        # set the positions for each of the data label
        self.data_label.set_position(x, y, disp)
        self._xd, self._yd = self.data_label._xd, self.data_label._yd

        if self.axis_label:
            # find the label with the closest x data point to the target position and place the x-axis marker there'
            self.axis_label.set_position(self.data_label._xd, self.data_label._yd)

    def get_data_points(self):
        return self._xd, self._yd

    def update_positions(self):
        self.set_position(*self._position_args)


class ScatterMarker(MarkerArtist):
    """
    Places a marker on a scatter plot.
    """

    def __init__(
        self,
        axes: Axes,
        collection: PathCollection,
        xlabel: dict = None,
        ylabel: dict = None,
        scatterdot: dict = None,
        xline: dict = None,
        yline: dict = None,
        xlabel_formatter: Callable = None,
        ylabel_formatter: Callable = None,
        anchor: str = "center left",
    ):
        """
        Parameters:
        -----------


        """
        self.collection = collection
        artists = []
        self.axes = axes
        self._anchor = anchor
        self._idx = 0
        self.xlabel_formatter = ylabel_formatter
        self.ylabel_formatter = ylabel_formatter
        self.scatterdot = None
        self.xlabel = None
        self.ylabel = None
        self.xline = None
        self.yline = None

        artists = []

        # create xline object
        if xline:
            self.xline = AxisLabel(axes, xlabel, False, xline, False, False, xlabel_formatter, None)
            artists += self.xline._artists

        # data labels for each line
        if yline:
            self.yline = AxisLabel(axes, False, ylabel, False, yline, False, None, ylabel_formatter)
            artists += self.yline._artists

        # initalize marker dot at data point
        if scatterdot:
            # default color for the dot is same as the scatter plot
            if "color" not in scatterdot.keys():
                scatterdot["color"] = collection.get_facecolor()

            self.scatterdot = MarkerLine(axes=axes, **scatterdot)
            artists += [self.scatterdot]

        if ylabel and not yline:
            # match the ylabel color to the scatter collection color
            ylabel = dcopy(ylabel)
            ylabel["bbox"]["edgecolor"] = collection.get_facecolor()[0]
            self.ylabel = MarkerLabel(axes=axes, verticalalignment="bottom", **ylabel)
            artists += [self.ylabel]

        self.set_position(0, 0)

        super().__init__(axes, artists)

    def set_position(self, x: float, y: float, disp: bool = False):
        """
        Parameters:
        -----------
        """

        # save the arguments to update the marker position when the axes bbox changes later
        self._position_args = (x, y, disp)

        if disp:
            x, y = utils.display2data(self.axes, (x, y))

        xdata, ydata = self.collection.get_offsets().data.T

        if x is not None and y is not None:
            dist = np.sqrt((x - xdata) ** 2 + (y - ydata) ** 2)
        elif x is not None:
            dist = np.abs(x - xdata)
        elif y is not None:
            dist = np.abs(y - ydata)
        else:
            raise ValueError(f"Insufficent positional arguments for marker.")

        # set position to the data point with the smallest error
        self.set_position_by_index(np.nanargmin(dist))

    def set_position_by_index(self, idx):

        xdata, ydata = self.collection.get_offsets().data.T

        if idx >= len(xdata):
            idx = len(xdata) - 1

        if idx < 0:
            idx = 0

        self._idx = idx
        self._xd = np.real(xdata[idx])
        self._yd = np.real(ydata[idx])

        # get label position in display coordinates. Use the line axes instead of the class axes since
        # the line may belong to another twinx/y axes and have different scaling.
        xl, yl = utils.data2display(self.axes, (self._xd, self._yd))

        if self.ylabel:

            # pad values in display coordinates (pixels)
            label_pad = self.ylabel._offset_padding * (self.axes.figure.dpi / 100)

            ytxt = utils.label_formatter(
                self.axes, self._xd, self._yd, self._idx, mode="y", custom=self.ylabel_formatter,
            )

            self.ylabel.set_position(
                (xl, yl), ytxt, anchor=self._anchor, disp=True, offset=(label_pad, label_pad),
            )

        if self.scatterdot:
            self.scatterdot.set_data([self._xd], [self._yd])

        if self.xline:
            self.xline.set_position(x=self._xd)

        if self.yline:
            self.yline.set_position(y=self._yd)

    def get_data_points(self):
        return self._xd, self._yd

    def update_positions(self):
        self.set_position(*self._position_args)
