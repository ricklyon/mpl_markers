import unittest
import mpl_markers as mplm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from parameterized import parameterized
from test_variables import SHOW_INTERACTIVE

FIG_NAMES = (
    "test_nan_values.png",
    "test_out_of_bounds.png",
    "test_axlines_ignore.png",
    "test_set_xy.png",
    "test_unequal_xdata.png",
    "test_alias.png",
    "test_marker_properties.png",
    "test_handler.png",
    "test_axis_limits.png",
    "test_polar.png",
    "test_polar_nan.png",
)


class TestLineMarker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fig_dir = Path(__file__).parent / ".figures"
        cls.fig_dir.mkdir(exist_ok=True)

        cls.ref_fig_dir = Path(__file__).parent / "reference_figures"

    @parameterized.expand(FIG_NAMES)
    def test_zfigures(self, figname):
        """checks generated figures against references"""

        figdata = plt.imread(self.fig_dir / figname)
        refdata = plt.imread(self.ref_fig_dir / figname)

        if SHOW_INTERACTIVE:
            plt.show()
        np.testing.assert_array_almost_equal(figdata, refdata)

    def test_nan_values(self):
        """checks markers on lines with partial NaN data"""

        fig, ax = plt.subplots(1, 1)
        ax.set_title("test_nan_values")
        x1 = np.linspace(-np.pi, np.pi, 4000)

        d1 = np.sin(x1)
        d1[1900:2100] = np.nan

        ax.plot(x1, d1)
        ax.plot(x1, np.cos(x1))

        # place marker in data region with nan values
        m1 = mplm.line_marker(x=0)
        # marker outside of nan region
        m2 = mplm.line_marker(x=-2, axes=ax)

        # get list of data values for each line
        m1_xd, m1_yd = m1.get_data_points()
        m2_xd, m2_yd = m2.get_data_points()

        np.testing.assert_almost_equal(m1_xd, 0, decimal=3)
        np.testing.assert_almost_equal(m2_xd, -2, decimal=3)

        # first line of first marker should be nan
        np.testing.assert_equal(m1_yd[0], np.nan)
        # second line should be sin(0)
        np.testing.assert_almost_equal(m1_yd[1], 1, decimal=3)
        # both lines of second marker should be cos(-2)
        np.testing.assert_almost_equal(m2_yd[1], np.cos(-2), decimal=3)

        # verify that the first line marker is hidden since it is nan
        self.assertTrue(m1.data_labels[0].ylabel._hidden)
        # second line should be
        self.assertFalse(m1.data_labels[1].ylabel._hidden)

        fig.savefig(self.fig_dir / "test_nan_values.png")

    def test_out_of_bounds(self):
        """tests marker placement if set to a out of bounds x value"""

        fig, ax = plt.subplots(1, 1)
        ax.set_title("test_out_of_bounds")

        ax.plot(2 * np.arange(10))
        m1 = mplm.line_marker(x=-1)
        m2 = mplm.line_marker(x=11)

        m1_xd, m1_yd = m1.get_data_points()
        m2_xd, m2_yd = m2.get_data_points()

        np.testing.assert_equal(m1_xd, 0)
        np.testing.assert_equal(m2_xd, 9)
        np.testing.assert_equal(m1_yd, 0)
        np.testing.assert_equal(m2_yd, 18)

        fig.savefig(self.fig_dir / "test_out_of_bounds.png")

    def test_axlines_ignore(self):
        """checks that markers do not attach to vertical or horizontal axlines"""
        fig, ax = plt.subplots(1, 1)
        ax.set_title("test_axlines_ignore")

        ax.plot(2 * np.arange(10))
        ax.axhline(y=5)
        ax.axvline(x=6)

        m1 = mplm.line_marker(x=4)
        m1_xd, m1_yd = m1.get_data_points()

        self.assertEqual(len(m1_xd), 1)
        self.assertEqual(m1_xd[0], 4)
        self.assertEqual(m1_yd[0], 8)

        fig.savefig(self.fig_dir / "test_axlines_ignore.png")

    def test_set_xy(self):
        """sets a marker using both x and y data points"""
        fig, ax = plt.subplots(1, 1)
        ax.set_title("test_set_xy")

        ax.plot(20 * np.arange(10))
        m1 = mplm.line_marker(x=4, y=100)
        m1_xd, m1_yd = m1.get_data_points()

        # marker should snap to nearest x value since data is monotonic
        self.assertEqual(m1_yd[0], 80)
        self.assertEqual(m1_xd[0], 4)

        fig.savefig(self.fig_dir / "test_set_xy.png")

    def test_unequal_xdata(self):
        """checks that markers work if lines have unequal data lengths"""

        fig, ax = plt.subplots(1, 1)
        ax.set_title("test_unequal_xdata")
        x1 = np.arange(-2, 2, 0.1)
        x2 = np.arange(-4, 4, 0.5)

        ax.plot(x1, np.sin(x1), label="sin(x)")
        ax.plot(x2, np.cos(x2), label="cos(x)")

        m1 = mplm.line_marker(x=1)
        m1_xd, m1_yd = m1.get_data_points()

        np.testing.assert_almost_equal(m1_xd, 1)
        np.testing.assert_almost_equal(m1_yd[0], np.sin(1))
        np.testing.assert_almost_equal(m1_yd[1], np.cos(1))

        fig.savefig(self.fig_dir / "test_unequal_xdata.png")

    def test_alias(self):
        """use alias data to set the marker at a angle instead of xy point."""
        fig, (ax1) = plt.subplots(1, 1)
        ax1.set_title("test_alias")
        ax1.set_aspect("equal")
        x1 = np.linspace(0, 2 * np.pi, 1000)
        x1_deg = np.rad2deg(x1)

        d = np.exp(1j * x1)
        d[400:600] = np.nan
        ln1 = ax1.plot(np.real(d), np.imag(d))
        ln2 = ax1.plot(0.9 * np.real(d[:200]), 0.9 * np.imag(d[:200]))

        # marker shows the angle in the ylabel box instead of the y data, which in this case
        # would be the imaginary part.
        # marker is set on the first line only
        angle = 30
        m1 = mplm.line_marker(
            x=angle,
            alias_xdata=x1_deg,
            xline=False,
            yformatter=lambda x, y, idx: r"{:.2f}$^\circ$".format(x1_deg[idx]),
            lines=ln1,
        )

        m1_xd, m1_yd = m1.get_data_points()
        np.testing.assert_almost_equal(np.cos(np.deg2rad(angle)), m1_xd, decimal=2)
        np.testing.assert_almost_equal(np.sin(np.deg2rad(angle)), m1_yd, decimal=2)

        # check that only the first line was added to the marker
        self.assertEqual(len(m1_yd), 1)

        fig.savefig(self.fig_dir / "test_alias.png")

    def test_marker_properties(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_title("test_marker_properties")
        x2 = np.linspace(-3 * np.pi, 3 * np.pi, 400)

        ax.plot(x2, np.cos(x2), label="cos(x)")

        m = mplm.line_marker(
            x=0,
            ylabel=dict(
                fontfamily="monospace",
                color="teal",
                fontsize=8,
                bbox=dict(linewidth=0, facecolor="none"),
            ),
            xline=dict(linewidth=2, color="teal", alpha=0.2),
            xlabel=dict(fontfamily="monospace", color="teal", fontsize=8),
        )

        fig.savefig(self.fig_dir / "test_marker_properties.png")

    def test_handler(self):
        """check marker handlers, used to link markers across different axes"""
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title("test_handler")

        par1 = ax1.twinx()

        x1 = np.linspace(-3 * np.pi, 3 * np.pi, 2000)
        x2 = np.linspace(-1 * np.pi, 1 * np.pi, 2000)

        ax1.plot(x1, np.sin(x1), label="sin(x)", color="c")
        par1.plot(x1, 2 * np.cos(x1), label="sin(x)", color="y")
        ax2.plot(x2, np.cos(x2), label="cos(x)", color="m")

        # create handlers that synchronize the active marker on each axes, called whenever markers are created/moved
        # xd and yd are arrays of the data values on each line, use the first line data to synchronize
        def link_ax1(xd, yd, **kwargs):
            mplm.move_active(x=xd[0], axes=ax2)

        def link_ax2(xd, yd, **kwargs):
            mplm.move_active(x=xd[0], axes=ax1)

        mplm.init_axes(ax1, handler=link_ax1)
        mplm.init_axes(ax2, handler=link_ax2)

        m1 = mplm.line_marker(x=1, axes=ax1, xlabel=True)
        m2 = mplm.line_marker(x=0, axes=ax2, xlabel=True)

        # move ax1 marker and make sure they stay synced on ax2
        mplm.move_active(x=2, call_handler=True, axes=ax1)

        # first axes should have two lines, including the twinx axes
        m1_xd, m1_yd = m1.get_data_points()
        m2_xd, m2_yd = m2.get_data_points()
        self.assertEqual(len(m1_xd), 2)
        self.assertEqual(len(m2_xd), 1)

        np.testing.assert_array_almost_equal(m1_xd, 2, decimal=2)
        np.testing.assert_array_almost_equal(m2_xd, 2, decimal=2)
        np.testing.assert_array_almost_equal(m1_yd[0], np.sin(2), decimal=2)
        np.testing.assert_array_almost_equal(m1_yd[1], 2 * np.cos(2), decimal=2)
        np.testing.assert_array_almost_equal(m2_yd[0], np.cos(2), decimal=2)

        fig.savefig(self.fig_dir / "test_handler.png")

    def test_axis_limits(self):
        """checks that axis limits are not modified by marker objects"""
        fig, ax = plt.subplots(1, 1)
        ax.set_title("test_axis_limits")
        x1 = np.linspace(np.pi, 2 * np.pi, 1000)

        ax.plot(x1, np.cos(x1), label="cos(x)")
        ax.plot(x1, np.sin(x1), label="sin(x)")
        ax.legend(loc="lower left")
        plt.margins(x=0)

        m1 = mplm.line_marker(x=-2 * np.pi)
        self.assertEqual(ax.get_xlim(), (np.pi, 2 * np.pi))

        fig.savefig(self.fig_dir / "test_axis_limits.png")

    def test_polar(self):
        """test polar markers"""
        fig, (ax1) = plt.subplots(1, 1, subplot_kw={"projection": "polar"})
        ax1.set_title("test_polar")
        x2 = np.linspace(-np.pi, np.pi, 1000)

        ax1.plot(x2, np.cos(x2) ** 2)
        ax1.plot(x2, np.cos(x2))

        m = mplm.line_marker(x=np.pi / 3)

        m_x, m_y = m.get_data_points()
        np.testing.assert_array_almost_equal(m_x, np.pi / 3, decimal=2)
        np.testing.assert_array_almost_equal(m_y[0], np.cos(np.pi / 3) ** 2, decimal=2)
        np.testing.assert_array_almost_equal(m_y[1], np.cos(np.pi / 3), decimal=2)

        fig.savefig(self.fig_dir / "test_polar.png")

    def test_polar_nan(self):
        """tests polar markers on data that is below the axes y limit"""
        fig, ax1 = plt.subplots(1, 1, subplot_kw=dict(polar=True))

        ax1.set_title("test_polar_nan")
        theta = np.linspace(-180, 180, 200)
        theta_rad = np.deg2rad(theta)

        ax1.plot(theta_rad, np.abs(np.sin(theta_rad)))
        ax1.set_ylim([0.5, 1])

        # second marker should be at sin(1/10) ~ .3, which is NaN since it is below the axis limit
        m1 = mplm.line_marker(x=-np.pi / 2)
        m2 = mplm.line_marker(x=np.pi / 10)

        m1_x, m1_y = m1.get_data_points()
        m2_x, m2_y = m2.get_data_points()

        np.testing.assert_array_almost_equal(m1_x, -np.pi / 2, decimal=2)
        np.testing.assert_array_almost_equal(m2_x, np.pi / 10, decimal=2)

        np.testing.assert_array_almost_equal(m1_y, 1, decimal=2)
        np.testing.assert_array_almost_equal(
            m2_x, np.abs(np.sin(np.pi / 10)), decimal=2
        )

        fig.savefig(self.fig_dir / "test_polar_nan.png")


if __name__ == "__main__":
    unittest.main()
