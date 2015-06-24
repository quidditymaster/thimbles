import threading
import numpy as np
import matplotlib as mpl

from thimblesgui import QtCore, QtGui, Qt

from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from thimbles.charts import MatplotlibCanvas


class MatplotlibWidget(QtGui.QWidget):
    """
    Matplotlib widget
    
    Attributes
    ----------
    canvas : the MplCanvas object, which contains the figure, axes, etc.
    axes : the main axe object for the figure
    vboxlayout : a vertical box layout from QtGui
    mpl_toolbar : boolean
      whether the widget should have a regular matplotlib toolbar or not.
    
    Notes
    -----
    __1)__ S. Tosi, ``Matplotlib for Python Developers''
        Ed. Packt Publishing
        http://www.packtpub.com/matplotlib-python-development/book
        
    """
    buttonPressed = QtCore.Signal(list)#mpl.backend_bases.Event)
    buttonReleased = QtCore.Signal(list)#mpl.backend_bases.Event)
    pickEvent = QtCore.Signal(list)#mpl.backend_bases.Event)
    
    def __init__(
            self, 
            nrows=1, 
            ncols=1, 
            mpl_toolbar=True,
            sharex="none", 
            sharey="none", 
            parent=None,
            fig_kws=None,
            redraw_framerate=10,
    ):
        #self.parent = parent
        # initialization of Qt MainWindow widget
        super(MatplotlibWidget, self).__init__(parent)
        
        # set the canvas to the Matplotlib widget
        self.canvas = MatplotlibCanvas(nrows, ncols, sharex=sharex, sharey=sharey)
        self.fig = self.canvas.fig
        
        # create a vertical box layout
        self.vboxlayout = QtGui.QVBoxLayout()
        self.vboxlayout.addWidget(self.canvas)
        if mpl_toolbar:
            self.mpl_toolbar = NavigationToolbar(self.canvas,self)
            self.mpl_toolbar.setWindowTitle("Plot")
            #import thimblesgui at the beginning of your script to fix signature/subclassing bugs.
            self.vboxlayout.addWidget(self.mpl_toolbar)
        
        # set the layout to the vertical box
        self.setLayout(self.vboxlayout)
        
        self._mpl_qt_connect()
        
        self.redraw_timer = QtCore.QTimer()
        self.redraw_timer.timeout.connect(self.check_redraw)
        n_msec = max(int(1000/redraw_framerate), 20)
        self.redraw_timer.start(n_msec)
    
    def _mpl_qt_connect(self):
        #print "mpl connect called"
        #import pdb; pdb.set_trace()
        self.canvas.callbacks.connect("button_press_event", self.emit_button_pressed)
        self.canvas.callbacks.connect("button_release_event", self.emit_button_released)
        self.canvas.callbacks.connect("pick_event", self.emit_pick_event)
    
    def check_redraw(self):
        if self.canvas.figure._tmb_redraw:
            self.canvas.draw()
            self.canvas.figure._tmb_redraw=False
    
    def emit_button_pressed(self, event):
        self.buttonPressed.emit([event])
    
    def emit_button_released(self, event):
        self.buttonReleased.emit([event])
    
    def emit_pick_event(self, event):
        self.pickEvent.emit([event])
    
    @property
    def ax(self):
        return self.canvas.ax
    
    def set_ax(self, row, column):
        self.canvas.set_ax(row, column)
    
    def axis(self, row, column):
        return self.canvas.axis(row, column)
    
    def draw (self):
        self.canvas.draw()
