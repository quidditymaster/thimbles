import threading
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
try: 
    from PySide.QtCore import *
    from PySide.QtGui import *
    matplotlib.rcParams['backend.qt4'] = 'PySide'
except ImportError:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    matplotlib.rcParams['backend.qt4'] = 'PyQt4'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

import thimbles as tmb


class SpectrumWidget(QWidget):
    
    def __init__(self, spectrum):
        super(SpectrumWidget, self).__init__()
        #self.frame = QWidget()
        self.spec = spectrum
        self.dpi = 100
        self.fig = Figure((10.0, 6.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        
        self.axes = self.fig.add_subplot(111)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.canvas)
        self.vbox.addWidget(self.mpl_toolbar)
        self.setLayout(self.vbox)
        self._init_plots()
        self.draw()
    
    def _init_plots(self):
        self.datal ,= self.axes.plot(self.spec.wv, self.spec.flux, c="b")
        self.contl ,= self.axes.plot(self.spec.wv, self.spec.norm)
        self.maskl ,= self.axes.plot(self.spec.wv, self.spec.feature_mask*self.spec.norm)
    
    def update_flux(self):
        self.datal.set_ydata(self.spec.flux)
    
    def draw(self):
        self.canvas.draw()


class FloatSlider(QWidget):
    
    def __init__(self, name, min_, max_, n_steps=100, orientation=Qt.Horizontal, parent=None):
        super(FloatSlider, self).__init__(parent)
        label = QLabel(name, parent=self)
        if orientation == Qt.Horizontal:
            lay = QHBoxLayout()
        elif orientation == Qt.Vertical:
            lay = QVBoxLayout()
        else:
            raise NotImplementedError
        self.min = min_
        self.max = max_
        self.delta = float(max_-min_)
        self.slider = QSlider(orientation, self)
        self.n_steps = n_steps
        self.slider.setRange(0, n_steps)
        lay.addWidget(label)
        lay.addWidget(self.slider)
        self.setLayout(lay)

    def value(self):
        return self.min + self.slider.value()*(self.delta/self.n_steps)
    
    def set_value(self, val):
        vfrac = (val-self.min)/self.delta
        opt_idx = int(np.around(vfrac*self.n_steps))
        self.slider.setValue(opt_idx)


class MatplotlibCanvas (FigureCanvas):
    """
    Class to represent the FigureCanvas widget
    
    Attributes
    ----------
    fig : the handler for the matplotlib figure
    axes : the main axe object for the figure
    
    Methods
    -------
    
    Notes
    -----
    __1)__ adapted from S. Tosi, ``Matplotlib for Python Developers''
        Ed. Packt Publishing
        http://www.packtpub.com/matplotlib-python-development/book
    
    """
    
    def __init__(self):
        # setup Matplotlib Figure and Axis
        self.fig = Figure()
        super(MatplotlibCanvas,self).__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
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

class MatplotlibWidget(QWidget):
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
    __1)__ adapted from S. Tosi, ``Matplotlib for Python Developers''
        Ed. Packt Publishing
        http://www.packtpub.com/matplotlib-python-development/book
         
    """
    def __init__(self, parent=None, canvas=None):
        #self.parent = parent
        # initialization of Qt MainWindow widget
        QWidget.__init__(self, parent)
        
        # set the canvas to the Matplotlib widget
        if canvas==None:
            canvas = MatplotlibCanvas()
        self.canvas = canvas
        
        self.fig = canvas.fig
        self.ax = self.canvas.ax
        
        # create a vertical box layout
        self.vboxlayout = QVBoxLayout()
        self.vboxlayout.addWidget(self.canvas)
        self._init_toolbar()
        
        # set the layout to the vertical box
        self.setLayout(self.vboxlayout)
    
    def _init_toolbar (self):
        self.mpl_toolbar = NavigationToolbar(self.canvas,self)
        self.mpl_toolbar.setWindowTitle("Plot")
        self.vboxlayout.addWidget(self.mpl_toolbar)
      
    def draw (self):
        self.canvas.draw()

class FeatureFitWidget(QWidget):
    
    def __init__(self, spectrum, features, feature_idx, display_width=10, parent=None):
        super(FeatureFitWidget, self).__init__(parent)
        
        self.display_width = display_width
        self.spectrum = spectrum
        self.features = features
        self.feature = features[feature_idx]
        self.feature_idx = feature_idx
        
        layout = QHBoxLayout()
        
        self.mpl_wid = MatplotlibWidget(parent=parent)
        self.ax = self.mpl_wid.ax
        layout.addWidget(self.mpl_wid)
        slider_orientation = Qt.Vertical
        self.off_slider = FloatSlider("offset", -0.1, 0.1, orientation=slider_orientation)
        self.d_slider = FloatSlider("depth", 0.0, 1.0, orientation=slider_orientation)
        self.g_slider = FloatSlider("sigma", 0.0, 1.0, orientation=slider_orientation)
        self.l_slider = FloatSlider("gamma", 0.0, 1.0, orientation=slider_orientation)
        for w in [self.off_slider, self.d_slider, self.g_slider, self.l_slider]:
            layout.addWidget(w)
        self._init_plots()
        self._init_slider_vals()
        self._connect_sliders()
        self.setLayout(layout)
    
    def bounded_spec(self):
        feat_wv = self.feature.wv
        min_wv = feat_wv-1.5*self.display_width
        max_wv = feat_wv+1.5*self.display_width
        bspec = self.spectrum.bounded_sample((min_wv, max_wv))
        return bspec
    
    def set_feature(self, feature_idx):
        self.feature_idx = feature_idx
        self.feature = self.features[feature_idx]
        self.on_feature_change()
    
    def sliders_changed(self, intval):
        #just ignore which slider caused the change get everything
        off = self.off_slider.value()
        gw = self.g_slider.value()
        lw = self.l_slider.value()
        depth = self.d_slider.value()
        self.feature.profile.set_parameters(np.array([off, gw, lw]))
        self.feature.set_depth(depth)
        self.update_plots()
    
    def on_feature_change(self):
        self._init_slider_vals()
        feat_wv = self.feature.wv
        xlim_min = feat_wv-self.display_width
        xlim_max = feat_wv+self.display_width
        self.ax.set_xlim(xlim_min, xlim_max)
        self.update_plots()
    
    def _connect_sliders(self):
        self.off_slider.slider.valueChanged.connect(self.sliders_changed)
        self.g_slider.slider.valueChanged.connect(self.sliders_changed)
        self.l_slider.slider.valueChanged.connect(self.sliders_changed)
        self.d_slider.slider.valueChanged.connect(self.sliders_changed)
    
    def _init_slider_vals(self):
        off, gw, lw = self.feature.profile.get_parameters()
        self.off_slider.set_value(off)
        self.g_slider.set_value(gw)
        self.l_slider.set_value(lw)
        d = self.feature.depth
        self.d_slider.set_value(d)
    
    def _init_plots(self):
        feat_wv = self.feature.wv
        xlim_min = feat_wv-self.display_width
        xlim_max = feat_wv+self.display_width
        self.ax.set_xlim(xlim_min, xlim_max)
        bspec = self.bounded_spec()
        self.data_line ,= self.ax.plot(bspec.wv, bspec.flux, c="b")
        self.cont_line ,= self.ax.plot(bspec.wv, bspec.norm, c="g")
        feature_model = self.feature.get_model_flux(bspec.wv)
        self.model_line,= self.ax.plot(bspec.wv, feature_model*bspec.norm)
        self.mpl_wid.canvas.draw()
    
    def update_plots(self):
        bspec = self.bounded_spec()
        self.data_line.set_data(bspec.wv, bspec.flux)
        bnorm = bspec.norm
        self.cont_line.set_data(bspec.wv, bnorm)
        feature_model = self.feature.get_model_flux(bspec.wv)
        self.model_line.set_data(bspec.wv, bnorm*feature_model)
        self.mpl_wid.draw()
