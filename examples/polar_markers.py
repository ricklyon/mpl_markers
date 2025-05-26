import numpy as np
import matplotlib.pyplot as plt

import mpl_markers as mplm

fig, (ax1) = plt.subplots(1, 1, subplot_kw={"projection": "polar"})
ax1.set_title("test_polar")
x2 = np.linspace(-np.pi, np.pi, 1000)

ax1.plot(x2, np.cos(x2) ** 2)
ax1.plot(x2, np.cos(x2))

m = mplm.line_marker(x=np.pi / 3, xlabel=True)

plt.show()

