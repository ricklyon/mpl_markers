import unittest
import mpl_markers as mplm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from parameterized import parameterized
from test_variables import SHOW_INTERACTIVE

FIG_NAMES = ("test_simple_scatter.png",)


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

    def test_simple_scatter(self):

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_title("test_simple_scatter")

        data_x = np.arange(10)
        data_y = data_x**2

        s = ax.scatter(data_x, data_y)
        mplm.scatter_marker(
            4,
            15,
            collection=s,
            axes=ax,
            scatterdot=dict(color="pink", markeredgewidth=0, markersize=15),
        )

        if SHOW_INTERACTIVE:
            plt.show()

        fig.savefig(self.fig_dir / "test_simple_scatter.png")


if __name__ == "__main__":
    unittest.main()
