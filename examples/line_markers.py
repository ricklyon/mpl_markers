import numpy as np
import matplotlib.pyplot as plt

import mpl_markers as mplm
from mpl_markers import utils, artists
import itertools
from mpl_markers.artists import LineMarker
from mpl_markers.utils import get_artist_bbox

mplm.set_style(
    ylabel=dict(fontfamily="monospace", bbox=dict(linewidth=0, facecolor="none"))
)


fig, ax = plt.subplots(1,1)
x1 = np.linspace(-np.pi/2, np.pi/2, 1000)

ax.plot(x1, np.sin(x1)*np.cos(x1)**2)

m = mplm.line_marker(idx = np.arange(0, 1000, 100))

plt.show()

