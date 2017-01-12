import time
import threading

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MatplotlibCanvas(FigureCanvas):
    """
    Class to represent the FigureCanvas widget
    
    nrows: int
      number of rows
    ncols: int
      number of columns
    sharex: [axis instance | "none" | "rows" | "columns" | "all"]
      determines which plots in the grid share x axes.
      axis instance : share x with given axis
      "none" no x axis sharing
      "rows" x axis shared by all plots in a row.
      "columns" x axis shared by all plots in a column
      "all" the x axis is shared between all plots
    sharey: ["none" | "rows" | "columns" | "all"]
      same as sharex for y axis sharing.
    
    """
    
    def __init__(
            self,
            nrows,
            ncols,
            sharex,
            sharey,
            projection=None,
            subplot_kws=None
    ):
        # setup Matplotlib Figure and Axis
        kws = dict(
            top=0.98,
            bottom=0.1,
            left=0.06,
            right=0.98,
        )
        if subplot_kws is None:
           subplot_kws = {}
        subplot_kws = subplot_kws.copy()
        for k in kws:
           subplot_kws.setdefault(k,kws[k])
        self.fig = Figure()
        self.fig._tmb_redraw = False
        self.fig.subplotpars.update(**subplot_kws)
        super(MatplotlibCanvas,self).__init__(self.fig)
        assert nrows >= 1
        assert ncols >= 1
        self.nrows = nrows
        self.ncols = ncols
        ax_num = 1
        self.axes = []
        #import pdb; pdb.set_trace()
        for col_idx in range(nrows):
            for row_idx in range(ncols):
                x_share_ax = None
                y_share_ax = None
                if not isinstance(sharex, str):
                    x_share_ax = sharex
                elif sharex == "none":
                    x_share_ax = None
                elif sharex == "rows":
                    if row_idx == 0:
                        x_share_ax = None
                    else:
                        x_share_ax = self.axes[-col_idx]
                elif sharex == "columns":
                    if col_idx == 0:
                        x_share_ax = None
                    else:
                        x_share_ax = self.axes[-row_idx*ncols]
                elif sharex == "all":
                    x_share_ax = self.axes[0]
                else:
                    raise Exception("don't recognize this sharex behavior")

                if not isinstance(sharey, str):
                    y_share_ax = sharey
                if sharey == "none":
                    y_share_ax = None
                elif sharey == "rows":
                    if col_idx == 0:
                        y_share_ax = None
                    else:
                        y_share_ax = self.axes[-col_idx]
                elif sharey == "columns":
                    if row_idx == 0:
                        y_share_ax = None
                    else:
                        y_share_ax = self.axes[-row_idx*ncols]
                elif sharey == "all":
                    y_share_ax = self.axes[0]
                else:
                    raise Exception("don't recognize this sharey behavior")
                self.axes.append(self.fig.add_subplot(nrows, ncols, ax_num, sharex=x_share_ax, sharey=y_share_ax, projection=projection))
                ax_num += 1
        
        #set the current axis to the first axis
        self.ax = self.axes[0]
        self._lock = threading.RLock()
    
    def axis(self, row_idx, col_idx):
        ax_num = self.ncols*row_idx + col_idx
        return self.axes[ax_num]
    
    def set_ax(self, row_idx, col_idx):
        """change which axis .ax refers to"""
        self.ax = self.axis(row_idx, col_idx)
    
    def draw(self):
        self.lock()
        super(MatplotlibCanvas, self).draw()
        self.unlock()
    
    def blit(self, bbox=None):
        self.lock()
        super(MatplotlibCanvas, self).blit(bbox)
        self.unlock()
    
    def lock(self):
        self._lock.acquire()
    
    def unlock(self):
        self._lock.release()


