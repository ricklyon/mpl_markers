import numpy as np
import matplotlib.pyplot as plt

import mpl_markers as mplm

mplm.set_style(
    ylabel=dict(fontfamily="monospace", bbox=dict(linewidth=0, facecolor="white", boxstyle="round4"))
)

fig, ax = plt.subplots(1,1)
fig.set_dpi(100)

x1 = np.linspace(-np.pi/2, np.pi/2, 1000)
ax.plot(x1, np.sin(x1)*np.cos(x1)**2)

# place markers every 100 points
m = mplm.line_marker(idx = np.arange(100, 1000, 100))

plt.show()

