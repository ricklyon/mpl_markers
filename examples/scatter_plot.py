import mpl_markers as mplm
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

data_x = np.random.normal(-1, 1, 100)
data_y = np.random.normal(-1, 1, 100)

fig, ax = plt.subplots(1, 1)
s = ax.scatter(data_x, data_y, color="b")

mplm.set_style(ylabel=dict(fontsize=10))

# place the marker on the point closest to x=0, y=0.5
# the collections argument is optional, if not provided the marker is placed on the first
# collection found on the axes
mplm.scatter_marker(
    0.5, 0.5, 
    collection=s, 
    axes=ax, 
    yformatter=lambda x, y, pos: f"x={x:.2f}\ny={y:.2f}",
    anchor="upper left"
)

plt.show()

