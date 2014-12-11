import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import numpy as np
from thimbles.charts import MatplotlibCanvas


class ForkDiagram(object):
    _plots_initialized = False
    
    def __init__(self, 
                 xvals=None, 
                 depths=None, 
                 curve=None, 
                 nub_height=0.01, 
                 handle_height=0.05, 
                 spread_height=0.05, 
                 handle_locator=None, 
                 text="", 
                 ax=None,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax
        self.fig = ax.figure
        if handle_locator is None:
            handle_locator = np.mean
        if curve is None:
            curve = lambda x: np.ones(x.shape)
        self.depths=depths
        self.curve = curve
        self.nub_height=nub_height
        self.handle_height=handle_height
        self.spread_height=spread_height
        self.handle_locator = handle_locator
        self.text=text
        self.set_xvals(xvals)
    
    def _initialize_plots(self):
        if not self._plots_initialized:
            self._init_handle()
            self._init_tines()
            self._init_annotation()
            self.ax.add_line(self.handle)
            self.ax.add_collection(self.tines)
        self._plots_initialized = True
    
    def _init_handle(self):
        x = self.handle_x
        bot, top = self.handle_bottom, self.handle_top
        self.handle = mpl.lines.Line2D([x, x], [bot, top])
    
    def update_handle(self):
        x = self.handle_x
        bot, top = self.handle_bottom, self.handle_top
        self.handle.set_data([x, x], [bot, top])
    
    @property
    def handle_x(self):
        return self.handle_locator(self.xvals)
    
    @property
    def rel_handle_top(self):
        return self.rel_handle_bottom + self.handle_height
    
    @property
    def rel_handle_bottom(self):
        return 1.0 + self.spread_height + self.nub_height
    
    @property
    def handle_bottom(self):
        return self.curve(self.handle_x)*self.rel_handle_bottom
    
    @property
    def handle_top(self):
        return self.curve(self.handle_x)*self.rel_handle_top
    
    @property
    def nub_tops(self):
        return self.curve(self.xvals)*(1.0+self.nub_height)
    
    @property
    def tine_bottoms(self):
        return self.curve(self.xvals)*self.depths
    
    def _calc_tine_data(self):
        lvals = np.zeros((len(self.xvals), 3, 2))
        lvals[:, :2, 0] = self.xvals.reshape((-1, 1)) * np.ones((1, 2))
        lvals[:, 2, 0] = self.handle_x
        lvals[:, 0, 1] = self.tine_bottoms
        lvals[:, 1, 1] = self.nub_tops
        lvals[:, 2, 1] = self.handle_bottom
        return lvals
    
    def _init_tines(self):
        tine_data = self._calc_tine_data()
        self.tines = mpl.collections.LineCollection(tine_data)
    
    def update_tines(self):
        tine_data = self._calc_tine_data()
        self.tines.set_segments(tine_data)
    
    def _init_annotation(self):
        anot_xy = (self.handle_x, self.handle_top)
        self.annotation = mpl.text.Annotation(self.text, anot_xy)
        self.ax.add_artist(self.annotation)
    
    def update_annotation(self):
        self.annotation.set_text(self.text)
        self.annotation.set_x(self.handle_x)
        self.annotation.set_y(self.handle_top)
    
    def update(self):
        self.update_tines()
        self.update_handle()
        self.update_annotation()
    
    def set_xvals(self, xvals, update=True):
        if not xvals is None:
            self.xvals = np.asarray(xvals)
            if self.depths is None:
                self.depths = np.zeros(xvals.shape)
            elif self.depths.shape != self.xvals.shape:
                self.depths = np.zeros(xvals.shape)
            self._initialize_plots()
            if update:
                self.update()
    
    def set_curve(self, curve, update=True):
        self.curve = curve
        if update:
            self.update()
    
    def set_depths(self, depths, update=True):
        self.depths = depths
        if update:
            self.update()
    
