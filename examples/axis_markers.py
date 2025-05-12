import mpl_markers as mplm
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
y1 = np.random.normal(6, 3, size=10)
ax.bar(np.arange(10), y1)
ax.margins(x=0.2)

# create horizontal axis marker
m1 = mplm.axis_marker(
    y=np.min(y1), yformatter="{:.2f}%", ylabel=dict(fontfamily="monospace"), placement="upper"
)

# create second marker that is referenced from the first marker m1
mplm.axis_marker(
    y=np.max(y1),
    ref_marker=m1,
    yformatter="{:.2f}%",
    ylabel=dict(fontfamily="monospace"),
    placement="upper"
)

plt.show()
