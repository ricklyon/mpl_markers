import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from copy import deepcopy as dcopy
from typing import Tuple, Union, Callable
import itertools
import matplotlib

from . import utils


class AbstractArtist(object):
    def __init__(self, axes=None): 

        self.axes = axes
        self.set_visible(False)
        self._hidden = False

    def draw_artist(self, renderer=None):
        """Draw each artist associated with marker."""

        if self._hidden:
            return 

        elif renderer:
            # super calls the next method in the MRO, which should be Text or Line2D
            super().draw(renderer)

        else:
            self.set_visible(True)
            self.axes.draw_artist(self)
            self.set_visible(False)

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

    def __init__(self, axes=None, **kwargs):
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

        # this calls the __init__ method for the class in the MRO after AbstractArtist, so mpl.Text
        # let the label be visible outside the axes for polar plots with clip_on=False
        super(AbstractArtist, self).__init__(0, 0, "0", clip_on=False, **kwargs)
        # call the AbstractArtist init method
        super(MarkerLabel, self).__init__(axes)

        # get axes from kwargs and add to axes
        axes.add_artist(self)

    def set_position(self, point:Tuple[float, float], text:str=None, anchor: str='upper left', disp: bool=False, offset: Tuple[float, float]=None):
        """
        Set label position and text

        archor kwarg specifies where the placement point is on the label:
        'upper/lower/center left/right/center'
        """
        offset = (0,0) if offset is None else offset
        # set text first so label is sized correctly
        if text is not None:
            self.set_text(text)

        # convert to display coordinates
        point = utils.data2display(self.axes, point) if not disp else point

        # get the label size
        xy0, xy1 = utils.get_artist_bbox(self)
        xlen, ylen = xy1 - xy0

        # set_position places the label's lower left corner at the point. If the user specifies the placement point 
        # at 'upper center' for example, the placement needs to be offset by x=+xlen and y=+ylen/2 to put the lower center 
        # of the label at the 'point' argument location.
        # matplotlib display coordinate system is (0,0) at lower left and (width, height) at the upper right
        # https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html

        # first word in loc is vertical position, second is horizontal
        loc_v, loc_h = anchor.split()
        v_offset = {'upper':-ylen, 'center':-ylen/2, 'lower':0}[loc_v]
        h_offset = {'left':0, 'center':-xlen/2, 'right':-xlen}[loc_h]

        # apply correction to find the position of the lower left corner
        b_left, b_lower  = np.array(point) + np.array([h_offset, v_offset]) + np.array(offset)

        # build bbox positions for a label placed at the upper left point
        l_bbox = np.array([[b_left, b_lower],
                           [b_left + xlen,  b_lower + ylen]])

        # force label within axes bounds
        ax_pad = self.axes.figure.dpi / 15
        # make axes smaller with negative padding
        ax_bbox = utils.get_artist_bbox(self.axes, (-ax_pad, -ax_pad))
        
        # arrays to subtract label bbox from axes bbox
        ax_sub = np.array([[1, 1], [-1, -1]])
        l_sub =  np.array([[-1, -1], [1, 1]])
        # distance that each point extrudes past the axes limits. If the label is fully within axes, the 
        # value will be negative and clip will force it to 0.
        lowerleft_ex, upperright_ex = np.clip((l_bbox * l_sub) + (ax_bbox * ax_sub), 0, None)

        if np.any(offset):
            # if user specified an offset, mirror around the placement point to attempt to bring the label back in bounds
            l_bbox += (np.array([xlen, ylen]) + 2*np.array(offset)) * (1*(lowerleft_ex > 0) - 1*(upperright_ex > 0)) 
            # recompute the extrusion distances
            lowerleft_ex, upperright_ex = np.clip((l_bbox * l_sub) + (ax_bbox * ax_sub), 0, None)

        # add extruded distances to label bbox to force within bounds
        l_bbox += lowerleft_ex
        l_bbox -= upperright_ex

        # use lower left bbox point for placement
        # AbstractArtist should not have a set_position method, so this will use set_position from mpl.Text
        super().set_position(l_bbox[0])


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
        self._dependent_markers = dependent_markers

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
            

    def set_hidden(self, state):
        [obj.set_hidden(state) for obj in self._artists if obj]

    def get_dependent_markers(self):
        return self._dependent_markers

    def add_dependent_marker(self, marker):
        self._dependent_markers.append(marker)


class LineLabel(MarkerArtist):
    """
    Text label attached to a data point of a Line2D object.
    """

    def __init__(self, 
                 axes: Axes, 
                 data_line: Line2D, 
                 xydot: dict = None, 
                 ylabel: dict = None, 
                 yline: dict = None, 
                 ylabel_formatter: Callable = None,
                 alias_xdata: np.ndarray = None
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
        self.xydot = None
        self.yline = None
        self.ylabel_formatter = ylabel_formatter
        self._alias_xdata = alias_xdata
        self.data_line = data_line
        
        if ylabel:
            # change the edge color of the text box to the line color
            ylabel = dcopy(ylabel)
            ylabel["bbox"]["edgecolor"] = data_line.get_color()

            # initalize text box as the label
            self.ylabel = MarkerLabel(axes=axes, transform=None, verticalalignment="bottom", **ylabel)

            # set text formatter
            if ylabel_formatter is None:
                self.ylabel_formatter = utils.yformatter
        
        if xydot:
            # initalize marker dot at data point
            self.xydot = MarkerLine(axes=data_line.axes, color=data_line.get_color(), **xydot)

        if yline:
            self.yline = MarkerLine(axes=data_line.axes, **yline)

        self._xdata = np.array(self.data_line.get_xdata())
        self._ydata = np.array(self.data_line.get_ydata())

        if axes.name == 'polar':
            # wrap xdata between -180 and 180 if axes is polar
            self._xdata = (self._xdata + np.pi) % (2 * np.pi) - np.pi

        # validate alias data
        if np.any(self._alias_xdata) and self._alias_xdata.shape != self._xdata.shape:
            raise ValueError('Invalid alias data shape: {}. Needed: {}'.format(self._alias_xdata.shape, self._xdata.shape))

        artists = [self.yline, self.xydot, self.ylabel]

        super().__init__(axes, artists)
        # set to arbitrary position to intialize member variables.
        self._set_position_by_index(idx=0)

    def _set_position_by_index(self, idx):
    
        self._xdata = np.array(self.data_line.get_xdata())
        self._ydata = np.array(self.data_line.get_ydata())

        if idx < 0 or idx >= len(self._xdata):
            raise ValueError(f'marker index placement of {idx} out of bounds for data with length {len(self._xdata)}.')
        
        self._idx = idx
        self._xd = np.real(self._xdata[idx])
        self._yd = np.real(self._ydata[idx])

        # pad values in display coordinates (pixels)
        label_xpad = 10 * self.axes.figure.dpi / 100 if not self.yline else 0

        #hide marker if data is not finite or NaN
        if not np.isfinite(self._yd) or not np.isfinite(self._xd):
            self.set_hidden(True)
            return
        else:
            self.set_hidden(False)

        # get label position in display coordinates. Use the line axes instead of the class axes since
        # the line may belong to another twinx/y axes and have different scaling.
        xl, yl = utils.data2display(self.data_line.axes, (self._xd, self._yd))

        # set label to left side of axes if yline is active
        xl = 0 if self.yline else xl
            
        if self.ylabel:
            # set the x position to the data point location plus a small pad (in display coordinates)
            xd_lbl = self._alias_xdata[idx] if np.any(self._alias_xdata) else self._xd 
            txt = self.ylabel_formatter(xd_lbl, self._yd, idx=self._idx, axes=self.axes)

            self.ylabel.set_position((xl, yl), txt, anchor='center left', disp=True, offset=(label_xpad, 0))
            # get the text label from the formatter
        
        if self.yline:
            self.yline.set_data(self.axes.get_xlim(), [self._yd]*2)

        if self.xydot:
            self.xydot.set_data([self._xd], [self._yd])

    def set_position(self, x: float=None, y: float=None, disp: bool=False, mode: str=None):
        """
        Returns the index of the line data for the point closest to the x,y data values, and the distance in data coordinates
        """
        if disp:
            x, y = utils.display2data(self.axes, (x, y))
        
        if self.axes.name == 'polar' and x is not None:
            x = (x + np.pi) % (2 * np.pi) - np.pi

        # ignore positional arguments based on the placement mode.
        if mode == 'x' and not disp and np.any(self._alias_xdata):
            dist = np.abs(x - self._alias_xdata)
        elif mode == 'x' and x is not None:
            dist = np.abs(x - self._xdata)
        elif mode == 'y' and y is not None:
            dist = np.abs(y - self._ydata)
        elif x is not None and y is not None: # placement mode 'xy' requires both arguments
            dist = np.abs(x - self._xdata) + np.abs(y - self._ydata)
        else:
            raise ValueError(f'Insufficent positional arguments for marker with placement mode: {mode}')
        
        # set position to the data point with the smallest error
        self._set_position_by_index(np.nanargmin(dist))



class AxisLabel(MarkerArtist):
    """
    Places markers on the x and y axes. Placement is not constrained to data points.
    """
    def __init__(self, 
                 axes: Axes, 
                 xlabel: dict = None, 
                 ylabel: dict = None, 
                 xline: dict = None, 
                 yline: dict = None, 
                 xymark: dict = None,
                 xlabel_formatter: Callable = None,
                 ylabel_formatter: Callable = None,
                 ref_marker : MarkerArtist = None
    ):
        
        self.xlabel = None
        self.ylabel = None
        self.xline = None
        self.yline = None
        self.xymark = None
        self.xlabel_formatter = xlabel_formatter
        self.ylabel_formatter = ylabel_formatter
        self.ref_marker = ref_marker

        if xline:
            self.xline = MarkerLine(axes=axes, **xline)

        if xlabel:
            self.xlabel = MarkerLabel(axes=axes, transform=None, verticalalignment="bottom", **xlabel)
            # set text formatter
            if xlabel_formatter is None:
                self.xlabel_formatter = utils.xformatter

        if yline:
            self.yline = MarkerLine(axes=axes, **yline)

        if ylabel:
            self.ylabel = MarkerLabel(axes=axes, transform=None, verticalalignment="bottom", **ylabel)
            # set text formatter
            if ylabel_formatter is None:
                self.ylabel_formatter = utils.yformatter

        if xymark and xline and yline:
            # initalize marker dot at data point
            self.xymark = MarkerLine(axes=axes, **xymark)

        self._xd = 0
        self._yd = 0

        artists = [self.xline, self.yline, self.xymark, self.xlabel, self.ylabel]
        super().__init__(axes, artists)

        self.set_position(0,0)

        if ref_marker:
            ref_marker.add_dependent_marker(self)

    def set_position(self, x:float = None, y:float = None, disp:bool = False, x_alias:float=None):

        self._position_args = (x, y, disp, x_alias)

        if disp:
            x, y = utils.display2data(self.axes, (x, y))
        
        # set x-axes marker
        if x is not None:
            # force within axes bounds
            
            if self.axes.name == 'polar':
                # wrap xdata between -180 and 180 if axes is polar
                x = (x + np.pi) % (2 * np.pi) - np.pi
            else:
                x = np.clip(x, *self.axes.get_xlim())

            self._xd = x if not x_alias else x_alias

            if self.xlabel:
                # use reference data if available
                if self.ref_marker:
                    lbl_xd = self._xd - self.ref_marker._xd
                    lbl_sgn = '$\Delta$ +' if lbl_xd > 0 else '$\Delta$ -'
                    lbl = lbl_sgn + self.xlabel_formatter(np.abs(lbl_xd), self._yd, axes=self.axes)
                else:
                    lbl = self.xlabel_formatter(self._xd, self._yd, axes=self.axes)

                xl, _ = utils.data2display(self.axes, (x, 0))
                self.xlabel.set_position((xl, 0), lbl, anchor='lower center', disp=True)

            if self.xline:
                self.xline.set_data([x]*2, self.axes.get_ylim())

        # set y-axes marker
        if y is not None:
            # force within axes bounds
            y = np.clip(y, *self.axes.get_ylim())
            self._yd = y

            _, yl = utils.data2display(self.axes, (0, y))

            if self.ylabel:
                # use reference data if available
                if self.ref_marker:
                    lbl_yd = self._yd - self.ref_marker._yd
                    lbl_sgn = '$\Delta$ +' if lbl_yd > 0 else '$\Delta -$'
                    lbl = lbl_sgn + self.ylabel_formatter(self._xd, np.abs(lbl_yd), axes=self.axes)
                else:
                    lbl = self.ylabel_formatter(self._xd, self._yd, axes=self.axes)

                self.ylabel.set_position((0, yl), lbl, anchor='center left', disp=True)

            if self.yline:
                self.yline.set_data(self.axes.get_xlim(), [y]*2)

        if x is not None and y is not None and self.xymark:
            self.xymark.set_data([x], [y])
        elif x is not None and self.xymark:
            self.xymark.set_data([x], self.axes.get_ylim()[0])
        elif y is not None and self.xymark:
            self.xymark.set_data(self.axes.get_xlim()[0], [y])

    def update_positions(self):
        self.set_position(*self._position_args)


class DataMarker(MarkerArtist):
    """
    Places a marker on each line in axes.
    """
    def __init__(self, 
                 axes: Axes, 
                 lines: list[Line2D],
                 xlabel: dict = None,
                 ylabel: dict = None,
                 xydot: dict = None,
                 xline: dict = None,
                 yline: dict = None,
                 xlabel_formatter: Callable = None,
                 ylabel_formatter: Callable = None,
                 alias_xdata: np.ndarray = None
    ):
        """
        Parameters:
        -----------


        """
        self.data_labels = []
        artists = []
        self.xaxis_label = None
        self.lines = lines
        self.ylabel_artists = []
        self.xlabel_artist = None
        self.axes = axes
        self._alias_xdata = alias_xdata

        # check if all lines have monotonic x-axis data
        self._monotonic_xdata = True
        for ln in lines:
            if self._monotonic_xdata:
                diff = np.diff(ln.get_xdata())
                self._monotonic_xdata = np.all(diff >= 0) or np.all(diff <= 0)

        # check that ylines have associated labels or dots
        if yline and not (ylabel or xydot):
            raise TypeError('y-axis lines require labels or marker dots.')
        
        # create single x-axes label shared by all the data markers
        if xline or xlabel:
            self.xaxis_label = AxisLabel(axes, xlabel, False, xline, False, False, xlabel_formatter, None)
        
        # data labels for each line
        if ylabel or xydot or yline:
            # turn off ylabel on data markers if yline is present. The axes label will be used as the data label.
            self.data_labels = [LineLabel(axes, ln, xydot, ylabel, yline, ylabel_formatter, alias_xdata) for ln in lines]

        # build list of all artists in the marker

        # line and dot artists first
        if self.xaxis_label:
            artists += self.xaxis_label._artists[:2]
        artists += list(itertools.chain.from_iterable([lbl._artists[:-1] for lbl in self.data_labels])) 

        # then label artists on top
        artists += list(itertools.chain.from_iterable([lbl._artists[-1:] for lbl in self.data_labels])) 
        if self.xaxis_label:
            artists += self.xaxis_label._artists[2:]

        # get list of all labels in the marker
        if ylabel:
            self.ylabel_artists += [lbl.ylabel for lbl in self.data_labels]
        if self.xaxis_label:
            self.xlabel_artist = self.xaxis_label.xlabel

        self._has_xlabel = self.xaxis_label and self.xaxis_label.xlabel

        self.set_position(0)

        super().__init__(axes, artists)

    def set_position(self, x:float = None, y:float = None, disp: bool=False, mode:str = 'x'):
        """
        Parameters:
        -----------
        mode: str -- ['x', 'y', 'xy']
            controls if line markers are placed by the x value ('x'), the y value ('y'), or by both ('xy').
        """

        # override the placement mode if lines aren't monotonic
        if not self._monotonic_xdata and x and y:
            mode = 'xy'

        use_alias = bool(mode == 'x' and not disp and np.any(self._alias_xdata))
        # save the arguments to update the marker position when the axes bbox changes later
        self._position_args = (x, y, disp, mode)

        # set the positions for each of the line labels
        xd_yd = np.zeros((2, len(self.lines)))
        self._xlbl = None

        for ii, lbl in enumerate(self.data_labels):
            lbl.set_position(x, y, disp, mode)
            xd_yd[:, ii] = lbl._xd, lbl._yd

        self._xd, self._yd = xd_yd

        if self.xaxis_label:

            if self.axes.name == 'polar':
                # wrap xdata between -180 and 180 if axes is polar
                x = (x + np.pi) % (2 * np.pi) - np.pi
                    
            if use_alias:
                a_idx = np.nanargmin(np.abs(x - self._alias_xdata))
                x = self.lines[0].get_xdata()[a_idx]

            # find the label with the closest x data point to the target position and place the x-axis marker there'
            nearest_lbl_idx = np.nanargmin(np.abs(self._xd - x))
            self._xlbl = self._xd[nearest_lbl_idx]

            if np.any(self._alias_xdata):
                x_idx = np.nanargmin(np.abs(self.lines[0].get_xdata() - self._xlbl))
                x_alias = self._alias_xdata[x_idx]
            else:
                x_alias = None

            self.xaxis_label.set_position(self._xlbl, x_alias=x_alias)

        self._space_labels()

    def get_data_points(self):
        return self._xd, self._yd

    def update_positions(self):
        self.set_position(*self._position_args)

    def _space_labels(self):

        ax_pad = self.axes.figure.dpi / 15
        # make axes smaller with negative padding
        ax_bbox = utils.get_artist_bbox(self.axes, (-ax_pad, -ax_pad))

        pad = np.array([4,4])
        vis_artists = [obj for obj in self.ylabel_artists if not obj._hidden]
        if not len(vis_artists):
            return

        sorted_labels = sorted(vis_artists, key=lambda x: utils.get_artist_bbox(x, pad)[0,1])

        s_bbox = np.array([utils.get_artist_bbox(lbl, pad) for lbl in sorted_labels])

        # get overlap that each box makes with the one above it. use roll to place the upper boxes in the place of its lower neighbor
        bbox_above = np.roll(s_bbox, shift=-1, axis=0)
        # get overlap that each box makes with the one below it. 
        bbox_below = np.roll(s_bbox, shift=1, axis=0)

        # use roll again to flip the edges so top edge of each box is subtracted from the bottom edge of the upper box.
        # this computes the overlap each box makes with the one above it. 
        # negative values indicate margin, positive values indicate overlap
        bbox_ovl_upper = s_bbox - np.roll(bbox_above, 1, axis=1)
        # this is invalid for the last (top) box since it doesn't have
        # an upper neighbor. Set to the 
        bbox_ovl_upper[-1] = -np.inf

        # overlap of each box with the one below it
        bbox_ovl_lower = -s_bbox + np.roll(bbox_below, 1, axis=1)

        # lower box overlap is with the xlabel if present
        if self._has_xlabel:
            bbox_ovl_lower[0] = -s_bbox[0] + np.roll(utils.get_artist_bbox(self.xaxis_label.xlabel, pad), 1, axis=0)
        # if there is no xlabel, use the bounding box for the axes
        else:
            bbox_ovl_lower[0] = -s_bbox[0] + ax_bbox[0]

        # start from bottom and push labels up to avoid the xlabel marker
        for ii in range(0, len(s_bbox)):

            # get the amount of overlap with the label below this one
            pos_ovl = np.clip(bbox_ovl_lower[ii, 0,1], 0, None)

            # move this label up if there is horizontal overlap. We know there is overlap if the distance
            # vectors between opposite corners are going different directions. 
            x0, x1 = bbox_ovl_lower[ii, :, 0]

            if np.sign(x0) != np.sign(x1):
                s_bbox[ii, :, 1] += pos_ovl
                bbox_ovl_lower[ii, :, 1] -= pos_ovl
                bbox_ovl_upper[ii, :, 1] += pos_ovl

                if ii>0:
                    bbox_ovl_upper[ii-1, :, 1] -= pos_ovl

                if ii < len(s_bbox)-1:
                    # Since we moved this box up, we have to adjust the overlap of the next box.
                    bbox_ovl_lower[ii+1, :, 1] += pos_ovl

        bbox_ovl_upper[-1] = s_bbox[-1] - ax_bbox[1]
        bbox_ovl_upper[-1, :, 0] = np.array([-1, 1])

        # push labels down vertically, starting from middle-1 (since we already moved the middle one up, the middle-1 label has
        # no overlap with the one above it).
        for ii in range(len(s_bbox)-1, -1, -1):

            # amount of overlap this label has with the one above it
            pos_ovl = np.clip(bbox_ovl_upper[ii, 1,1], 0, None)

            # move this label down if there is horizontal overlap
            x0, x1 = bbox_ovl_upper[ii, :, 0]
            if np.sign(x0) != np.sign(x1):

                s_bbox[ii, :, 1] -= pos_ovl
                bbox_ovl_upper[ii, :, 1] -= pos_ovl
                bbox_ovl_lower[ii, :, 1] += pos_ovl

                if ii < len(s_bbox) -1:
                    bbox_ovl_lower[ii+1, : 1] -= pos_ovl
            
                if ii > 0:
                    # Since we moved this box down down, we have to adjust the overlap of the next box.
                    bbox_ovl_upper[ii-1, :, 1] += pos_ovl


        for ii, s in enumerate(sorted_labels):
            s.set_position(s_bbox[ii, 0]+pad, disp=True, anchor='lower left')
