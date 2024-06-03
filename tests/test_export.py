import unittest
import mpl_markers as mplm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class TestExportMarkers(unittest.TestCase):

    def test_svg_pdf(self):
        fig_dir = Path(__file__).parent / ".figures"

        fig, (ax1) = plt.subplots(1, 1)
        x1 = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

        ax1.plot(x1, np.cos(x1), label="cos(x)")
        ax1.legend(loc="lower left")

        m1 = mplm.line_marker(x=-1.5 * np.pi, axes=ax1)
        m2 = mplm.axis_marker(y=0.5, axes=ax1)

        dir_ = Path(__file__).parent
        fig.savefig(fig_dir / "mpl_test.pdf")
        fig.savefig(fig_dir / "mpl_test.svg")


if __name__ == "__main__":
    unittest.main()
