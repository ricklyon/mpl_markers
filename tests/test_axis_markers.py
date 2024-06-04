import unittest
import mpl_markers as mplm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from parameterized import parameterized
from test_variables import SHOW_INTERACTIVE

FIG_NAMES = (
    "test_axis_markers.png",
    "test_axis_single.png",
    "test_axis_formatting.png",
)


class TestAxisMarker(unittest.TestCase):

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

    def test_axis_markers(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_title("test_axis_markers")

        ax.bar(np.arange(10), np.sin(np.arange(10)))
        m1 = mplm.axis_marker(x=2, y=1)

        fig.savefig(self.fig_dir / "test_axis_markers.png")

    def test_axis_single(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_title("test_axis_markers2")
        x1 = np.linspace(-2 * np.pi, 2 * np.pi, 41)

        d1 = np.sin(x1)

        ax.plot(x1, d1, label="sin(x)")
        ax.legend(loc="lower left")
        m1 = mplm.axis_marker(y=np.max(d1), yformatter="max={:.2f}")
        m1 = mplm.axis_marker(x=2, xformatter="max={:.2f}")

        fig.savefig(self.fig_dir / "test_axis_single.png")

    def test_axis_formatting(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_title("test_axis_formatting")
        ax.bar(np.arange(10), np.sin(np.arange(10)))

        mplm.init_axes(
            axes=ax,
            ylabel=dict(
                fontfamily="monospace",
                color="teal",
                fontsize=8,
                bbox=dict(linewidth=0, facecolor="none"),
            ),
            yformatter="{:.3f}\n",
        )

        m1 = mplm.axis_marker(
            y=np.sin(2),
        )

        fig.savefig(self.fig_dir / "test_axis_formatting.png")


if __name__ == "__main__":
    unittest.main()
