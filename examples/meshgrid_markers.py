import numpy as np
import matplotlib.pyplot as plt
import mpl_markers as mplm
from pathlib import Path

dir_ = Path(__file__).parent

# create example meshgrid data
xy = np.linspace(-1, 1, 200)
x, y = np.meshgrid(xy, xy)
z = np.sin(2 * x) ** 2 + np.cos(5 * y) ** 2

# plot the data with pcolormesh
fig = plt.figure(dpi=200)
ax = fig.subplots(1, 1)
m = ax.pcolormesh(x, y, z, vmin=0, vmax=2, cmap="jet")
plt.colorbar(m)

# add a data marker at a single x/y point on the plot. x/y is in data coordinates.
mplm.mesh_marker(x=0.75, y=0.25)
plt.show()
