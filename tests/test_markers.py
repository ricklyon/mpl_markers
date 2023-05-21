import unittest
import mpl_markers as mplm
import numpy as np
import matplotlib.pyplot as plt

class TestStringMethods(unittest.TestCase):

    # @unittest.skip
    def test_nan_values(self):
        fig, ax = plt.subplots(1,1)
        x1 = np.linspace(-2*np.pi, 2*np.pi, 1000)

        d1 = np.sin(x1)
        d1[400:500] = np.nan

        ax.plot(x1, d1, label='sin(x)')
        ax.plot(x1, np.cos(x1), label='cos(x)')
        ax.legend(loc='lower left')
        m = mplm.data_marker(x=-1)


        self.assertLess(np.max(m._xd + 1), 1e-3)

        self.assertFalse(np.isfinite(m._yd[0]))
        self.assertTrue(m.data_labels[0].ylabel._hidden)
        self.assertFalse(m.data_labels[1].ylabel._hidden)


    # @unittest.skip
    def test_unequal_xdata(self):
        fig, ax = plt.subplots(1,1)
        x1 = np.linspace(-2*np.pi, 2*np.pi, 1000)
        x2 = np.linspace(-3*np.pi, 3*np.pi, 400)

        ax.plot(x1, np.sin(x1), label='sin(x)')
        ax.plot(x2, np.cos(x2), label='cos(x)')

        angle = 2.5*np.pi
        m = mplm.data_marker(x=angle)
        m_x, m_y = m.get_data_points()


        plt.show()

    # @unittest.skip
    def test_non_monotonic(self):
        fig, (ax1) = plt.subplots(1, 1)
        x1 = np.linspace(0, 2*np.pi, 1000)
        x2 = np.linspace(0, np.pi/2, 2000)

        d = np.exp(1j*x1)
        d[700:750]  = np.nan
        d2 = np.exp(1j*x2)/2
        ax1.plot(np.real(d), np.imag(d))
        ax1.plot(np.real(d2), np.imag(d2))

        # set marker to angle. The xdata for the marker is the real component of d, so to set the angle we need to specify the index
        angle = np.pi/2
        m_idx = np.argmin(np.abs(x1 - angle))
        m = mplm.data_marker(x=0)#, ylabel_formatter=lambda x, y, idx, **kwargs: '{:.2f}$\pi$'.format(x1[idx]/np.pi))

        m_x, m_y = m.get_data_points()

        plt.show()

    # @unittest.skip
    def test_alias(self):
        fig, (ax1) = plt.subplots(1, 1)
        x1_pi = np.linspace(0, 2, 1000)
        x1 = x1_pi * np.pi
        # x2 = np.linspace(0, np.pi/2, 1000)

        d = np.exp(1j*x1)
        # d[700:750]  = np.nan

        ax1.plot(np.real(d), np.imag(d))

        # set marker to angle. The xdata for the marker is the real component of d, so to set the angle we need to specify the index
        angle = 3/4

        def yfmt(x, y, idx):
            return '{:.3f}$\pi$'.format(y)

        def xfmt(x, y, idx):
            return '{:.3f}$\pi$'.format(x)

        m = mplm.data_marker(x=angle, yline=True, alias_xdata=x1_pi, yformatter=yfmt, xformatter=xfmt)

        plt.show()

    # @unittest.skip
    def test_marker_properties(self):
        fig, ax = plt.subplots(1,1)
        x2 = np.linspace(-3*np.pi, 3*np.pi, 400)

        ax.plot(x2, np.cos(x2), label='cos(x)')

        m = mplm.data_marker(x=0, xlabel=True, 
                            ylabel=dict(fontfamily='monospace', rotation=-10, color='teal', fontsize=8), 
                            xline=dict(linewidth=2, color='teal', alpha=0.2)
                            )

        plt.show()


    # @unittest.skip
    def test_twinx_axes(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.grid(linewidth=0.5, linestyle='-') 

        par1 = ax1.twinx()
        par2 = ax2.twinx()

        x1 = np.linspace(-3*np.pi, 3*np.pi, 2000)
        x2 = np.linspace(-1*np.pi, 1*np.pi, 1000)

        ax1.axhline(y=0)
        par1.plot(x1, 2*np.cos(x1), label='sin(x)', color='y')
        ax1.plot(x1, np.sin(x1), label='sin(x)', color='c')
        ax2.plot(x2, np.cos(x2), label='cos(x)', color='m')


        def link_ax1(xd, yd, **kwargs):
            mplm.move_active(x=xd[0], axes=ax2)

        def link_ax2(xd, yd, **kwargs):
            mplm.move_active(x=xd[0], axes=ax1)

        mplm.init_axes(ax1, xlabel=dict(fontsize=6, color="black"), handler=link_ax1)
        mplm.init_axes(ax2, xlabel=dict(fontsize=6, color="black"), handler=link_ax2)

        m1 = mplm.data_marker(x=1, axes=ax1, xlabel=True)
        m2 = mplm.data_marker(x=0, axes=ax2, xlabel=True)

        # move ax1 marker and make sure they stay synced
        mplm.move_active(x=2, call_handler=True, axes=ax1)
        plt.show()

    # @unittest.skip
    def test_polar(self):
        fig, (ax1) = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
        x2 = np.linspace(-np.pi, np.pi, 1000)

        ax1.plot(x2, np.cos(x2)**2)
        ax1.plot(x2, np.cos(x2))

        m = mplm.data_marker(x=np.pi/3)

        m_x, m_y = m.get_data_points()
        plt.show()

    def test_polar2(self):
        fig, axes = plt.subplots(1,1, subplot_kw=dict(polar=True))

        theta = np.linspace(-180, 180, 200)
        theta_rad = np.deg2rad(theta)

        axes.plot(theta_rad, np.abs(np.sin(theta_rad)))

        m = mplm.data_marker(x=-np.pi/2)
        plt.show()

    # @unittest.skip
    def test_axis_markers(sef):
        fig, ax = plt.subplots(1,1)
        x1 = np.linspace(-2*np.pi, 2*np.pi, 1000)

        d1 = np.sin(x1)
        d1[400:500] = np.nan

        ax.plot(x1, d1, label='sin(x)')
        ax.plot(x1, np.cos(x1), label='cos(x)')
        ax.legend(loc='lower left')
        m1 = mplm.axis_marker(x=-1, y=0)

        m2 = mplm.axis_marker(x=2, y=0.5, ref_marker=m1, xymark=dict(markeredgecolor='black'))
        plt.show()

    def test_axis_markers2(sef):
        fig, ax = plt.subplots(1,1)
        x1 = np.linspace(-2*np.pi, 2*np.pi, 1000)

        d1 = np.sin(x1)
        d1[400:500] = np.nan

        ax.plot(x1, d1, label='sin(x)')
        ax.plot(x1, np.cos(x1), label='cos(x)')
        ax.legend(loc='lower left')
        m1 = mplm.axis_marker(y=0)

        plt.show()

if __name__ == '__main__':
    unittest.main()
    