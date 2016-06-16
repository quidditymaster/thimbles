import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.collections import LineCollection

from thimblesgui import QtCore, QtGui, Qt
import thimbles as tmb

class TransitionMarkerChart(object):
    
    def __init__(
            self,
            transition,
            locator_func,
            ax,
            core_kwargs=None,
            outline_kwargs=None,
            y_min=-5.0,
            y_max=5.0,
            zorder=3,
    ):
        self.locator_func = locator_func
        self.y_min = y_min
        self.y_max = y_max
        
        self.ax = ax
        
        if core_kwargs is None:
            core_kwargs = {}
        if outline_kwargs is None:
            outline_kwargs = {}    
        
        core_kwargs.setdefault("color", "y")
        core_kwargs.setdefault("lw", 2.0)
        
        outline_kwargs.setdefault("color", "gray")
        outline_kwargs.setdefault("lw", 4.0)
        
        self.core_line ,= self.ax.plot([0, 0], [y_min, y_max], zorder=zorder, **core_kwargs)
        self.outline_line ,= self.ax.plot([0, 0], [y_min, y_max], zorder=zorder-1, **outline_kwargs)
        
        self.set_transition(transition)
    
    def set_transition(self, transition):
        self.transition = transition
        if transition is None:
            self.core_line.set_visible(False)
            self.outline_line.set_visible(False)
        else:
            self.core_line.set_visible(True)
            self.outline_line.set_visible(True)
            x_loc = self.locator_func(transition)
            self.core_line.set_xdata([x_loc, x_loc])
            self.outline_line.set_xdata([x_loc, x_loc])
        self.ax.figure._tmb_redraw = True
