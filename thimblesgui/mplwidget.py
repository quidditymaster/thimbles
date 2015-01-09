import threading
import numpy as np
import matplotlib

from PySide import QtCore
from PySide import QtGui
from PySide.QtCore import Signal, Slot
Qt = QtCore.Qt

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
    mpl_toolbar : if instanciated with a withNavBar=True, then this attribute
        is the navigation toolbar object (from matplotlib), to allow
        exploration within the axes.
        
    Notes
    -----
    __1)__ S. Tosi, ``Matplotlib for Python Developers''
        Ed. Packt Publishing
        http://www.packtpub.com/matplotlib-python-development/book
        
    """
    buttonPressed = Signal(list)
    buttonReleased = Signal(list)
    pickEvent = Signal(list)
    
    def __init__(self, parent=None, nrows=1, ncols=1, mpl_toolbar=True,
                 sharex="none", sharey="none"):
        #self.parent = parent
        # initialization of Qt MainWindow widget
        QtGui.QWidget.__init__(self, parent)
        
        # set the canvas to the Matplotlib widget
        self.canvas = MatplotlibCanvas(nrows, ncols, sharex=sharex, sharey=sharey)
        self.fig = self.canvas.fig
        
        # create a vertical box layout
        self.vboxlayout = QtGui.QVBoxLayout()
        self.vboxlayout.addWidget(self.canvas)
        if mpl_toolbar:
            self.mpl_toolbar = NavigationToolbar(self.canvas,self)
            self.mpl_toolbar.setWindowTitle("Plot")
            self.vboxlayout.addWidget(self.mpl_toolbar)
        
        # set the layout to the vertical box
        self.setLayout(self.vboxlayout)
        
        self.mpl_connect()
    
    def mpl_connect(self):
        #print "mpl connect called"
        #import pdb; pdb.set_trace()
        self.canvas.callbacks.connect("button_press_event", self.emit_button_pressed)
        self.canvas.callbacks.connect("button_release_event", self.emit_button_released)
        self.canvas.callbacks.connect("pick_event", self.emit_pick_event)
    
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
