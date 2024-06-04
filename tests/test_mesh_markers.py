import unittest
import mpl_markers as mplm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from parameterized import parameterized
from test_variables import SHOW_INTERACTIVE

FIG_NAMES = ("test_mesh.png",)


class TestMeshMarkers(unittest.TestCase):

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

    def test_mesh(self):
        # create example meshgrid data
        xy = np.linspace(-1, 1, 200)
        x, y = np.meshgrid(xy, xy)
        z = np.sin(2 * x) ** 2 + np.cos(5 * y) ** 2

        # plot the data with pcolormesh
        fig = plt.figure()
        ax = fig.subplots(1, 1)
        m = ax.pcolormesh(x, y, z, vmin=0, vmax=2)

        # add a data marker at a single x/y point on the plot. x/y is in data coordinates.
        mplm.mesh_marker(x=0.75, y=0.25)

        fig.savefig(self.fig_dir / "test_mesh.png")


if __name__ == "__main__":
    unittest.main()
