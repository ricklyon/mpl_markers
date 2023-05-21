# mpl-markers

Interactive data markers for line plots in matplotlib

## Installation

```bash
pip install mpl-markers
```

## Usage

```python
import mpl_markers as mplm
```

Add a marker attached to plotted data lines:
```python
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rc('font', size=7)

fig, ax = plt.subplots(1,1)
x1 = np.linspace(-2*np.pi, 2*np.pi, 1000)

ax.plot(x1, np.sin(x1)*np.cos(x1)**2)

mplm.data_marker(x=0)
```
![example1](https://raw.githubusercontent.com/ricklyon/mpl_markers/main/docs/img/example1.gif)

Add an axis marker that moves freely on the canvas:
```python
ax.xaxis.set_major_formatter(lambda x, pos: '{:.2f}$\pi$'.format(x/np.pi))
mplm.axis_marker(x=0, y=-0.2)
```

![example2](https://raw.githubusercontent.com/ricklyon/mpl_markers/main/docs/img/example2.gif)

## License

mpl-markers is licensed under the MIT License.