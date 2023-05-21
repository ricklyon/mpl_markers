import mpl_markers as mplm
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rc('font', size=7)

fig, ax = plt.subplots(1,1)
x1 = np.linspace(-2*np.pi, 2*np.pi, 1000)
ax.plot(x1, np.sin(x1)**3)


# create axis marker that is free to move around canvas. The yline and xline 
# kwargs are optional, but in this case we want to make the lines black instead of
# using the default red color.
m1 = mplm.axis_marker(x=-2, y=-0.5)

# create second marker that is referenced from the first marker m1
mplm.axis_marker(x=2, y=0.5, ref_marker=m1)

# click and drag the markers around the axes
plt.show()