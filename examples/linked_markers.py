import mpl_markers as mplm
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rc("font", size=7)

fig, (ax1, ax2) = plt.subplots(2, 1)

par1 = ax1.twinx()
par2 = ax2.twinx()

x1 = np.linspace(-3 * np.pi, 3 * np.pi, 3000)
x2 = np.linspace(-1 * np.pi, 1 * np.pi, 1000)

par1.plot(x1, np.cos(x1), color="m")
ax1.plot(x1, np.sin(x1), color="c")
ax2.plot(x2, np.cos(x2), color="m")

# marker labels will inherit from the axis formatter
ax1.xaxis.set_major_formatter(lambda x, pos: "{:.2f}$\pi$".format(x / np.pi))
ax2.xaxis.set_major_formatter(lambda x, pos: "{:.2f}$\pi$".format(x / np.pi))

ax1.set_xticks([])
ax2.set_xticks([])


# these are called when the marker position changes. xd is a list of the x-data coordinates of each line on the axes.
# Call move_active on the other axes with the x-position of the first line to keep the two markers in sync.
def link_ax1(xd, yd, **kwargs):
    mplm.move_active(x=xd[0], axes=ax2)


def link_ax2(xd, yd, **kwargs):
    mplm.move_active(x=xd[0], axes=ax1)


# initialize each axes and specify a handler to call when the marker moves position.
mplm.init_axes(ax1, handler=link_ax1)
mplm.init_axes(ax2, handler=link_ax2)

# create markers on each axes
m1 = mplm.line_marker(x=1, axes=ax1, xlabel=True)
m2 = mplm.line_marker(x=1, axes=ax2, xlabel=True)

# click and drag markers on the axes
plt.show()
