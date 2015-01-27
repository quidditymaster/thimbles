import time
import threading

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MatplotlibCanvas(FigureCanvas):
    """
    Class to represent the FigureCanvas widget
    
    nrows: int
      number of rows
    ncols: int
      number of columns
    sharex: ["none" | "rows" | "columns" | "all"]
      determines which plots in the grid share x axes.
      "none" no x axis sharing
      "rows" x axis shared by all plots in a row.
      "columns" x axis shared by all plots in a column
      "all" the x axis is shared between all plots
    sharey: ["none" | "rows" | "columns" | "all"]
      same as sharex for y axis sharing.
    
    Attributes
    ----------
    fig : the handler for the matplotlib figure
    axes : the main axes object for the figure
    
    Methods
    -------
    
    Notes
    -----
    __1)__ S. Tosi, ``Matplotlib for Python Developers''
        Ed. Packt Publishing
        http://www.packtpub.com/matplotlib-python-development/book
    
    """
    
    def __init__(self, nrows, ncols, sharex, sharey):
        # setup Matplotlib Figure and Axis
        self.fig = Figure()
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
                if sharex == "none":
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
                    raise Exception("don't recognize sharex behavior")
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
                    raise Exception("don't recognize sharey behavior")
                self.axes.append(self.fig.add_subplot(nrows, ncols, ax_num, sharex=x_share_ax, sharey=y_share_ax))
                ax_num += 1
        
        #set the current axis to the first axis
        self.ax = self.axes[0]
        self._lock = threading.RLock()
        
        #import pdb; pdb.set_trace()
        #self.fig.add_subplot(111)
        #self.ax.plot([0, 20], [0, 20])
        
        # we define the widget as expandable
        #FigureCanvas.setSizePolicy(self,
        # QtGui.QSizePolicy.Expanding,
        # QtGui.QSizePolicy.Expanding)
        # notify the system of updated policy
        #self.updateGeometry()
    
    def axis(self, row_idx, col_idx):
        ax_num = self.ncols*row_idx + col_idx
        return self.axes[ax_num]
    
    def set_ax(self, row_idx, col_idx):
        """change which axis .ax refers to"""
        self.ax = self.axis(row_idx, col_idx)
    
    def draw(self):
        #print "in draw"
        self.lock()
        #print "in lock"
        super(MatplotlibCanvas, self).draw()
        #print "unlocking"
        self.unlock()
        #print "unlocked"
    
    def blit(self, bbox=None):
        self.lock()
        super(MatplotlibCanvas, self).blit(bbox)
        self.unlock()
    
    def lock(self):
        self._lock.acquire()
    
    def unlock(self):
        self._lock.release()
