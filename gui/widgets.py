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

  
class PrevNext(QWidget):
    prev = Signal()
    next = Signal()
    
    def __init__(self, duration=1, parent=None):
        super(PrevNext, self).__init__(parent)
        layout = QGridLayout()
        self.prev_btn = QPushButton("prev")
        self.next_btn = QPushButton("next")
        self.duration = int(duration*1000)
        self.duration_le = QLineEdit("%5.3f" % duration)
        self.timer = QTimer(self)
        self.timer.start(self.duration)
        self.paused = True
        self.play_toggle_btn = QPushButton("Play/Pause")
        
        #add to layout
        layout.addWidget(self.prev_btn, 0, 0, 1, 1)
        layout.addWidget(self.next_btn, 0, 1, 1, 1)
        layout.addWidget(self.duration_le, 1, 0, 1, 2)
        layout.addWidget(self.play_toggle_btn, 2, 0, 1, 2)
        self.setLayout(layout)
        
        #connect stuff
        self.duration_le.editingFinished.connect(self.set_duration)
        self.timer.timeout.connect(self.on_timeout)
        self.prev_btn.clicked.connect(self.emit_prev)
        self.next_btn.clicked.connect(self.emit_next)
        self.play_toggle_btn.clicked.connect(self.toggle_pause)
    
    def emit_next(self):
        self.next.emit()
    
    def emit_prev(self):
        self.prev.emit()
    
    def on_timeout(self):
        #print "timer went off"
        if not self.paused:
            if self.duration > 0:
                self.emit_next()
            elif self.duration < 0:
                self.emit_prev()
    
    def set_duration(self):
        try:
            duration_text = self.duration_le.text()
            new_duration = int(float(duration_text)*1000)
            if abs(new_duration) < 10:
                raise Exception("duration too small")
            new_duration_success = True
        except:
            print "could not recognize new duration reverting to old"
            new_duration_success = False
            self.duration_le.setText("%5.5f" % self.duration)
        if new_duration_success:
            self.duration = new_duration
            self.timer.setInterval(abs(new_duration))
    
    def toggle_pause(self):
        if self.paused:
            self.paused = False
        else:
            self.paused = True

class FeatureFitWidget(QWidget):
    slidersChanged = Signal(int)
    nextFeature = Signal()
    prevFeature = Signal()
    
    def __init__(self, spectrum, features, feature_idx, display_width, parent=None):
        super(FeatureFitWidget, self).__init__(parent)
        
        self.display_width = display_width
        self.spectrum = spectrum
        self.features = features
        self.feature = features[feature_idx]
        self.feature_idx = feature_idx
        
        layout = QGridLayout()
        
        self.mpl_wid = MatplotlibWidget(parent=parent)
        self.ax = self.mpl_wid.ax
        layout.addWidget(self.mpl_wid, 0, 0, 2, 1)
        slider_orientation = Qt.Vertical
        self.off_slider = FloatSlider("offset", -0.15, 0.15, orientation=slider_orientation)
        self.d_slider = FloatSlider("depth", 0.0, 1.0, orientation=slider_orientation)
        self.g_slider = FloatSlider("sigma", 0.0, 1.0, orientation=slider_orientation)
        self.l_slider = FloatSlider("gamma", 0.0, 1.0, orientation=slider_orientation)
        slider_grid = [(1, 1, 1, 1), (1, 2, 1, 1), (1, 3, 1, 1), (1, 4, 1, 1)]
        slider_list = [self.off_slider, self.d_slider, self.g_slider, self.l_slider]
        for sl_idx in range(len(slider_list)):
            layout.addWidget(slider_list[sl_idx], *slider_grid[sl_idx])
        self.prev_next = PrevNext(duration=1.0, parent=self)
        layout.addWidget(self.prev_next, 0, 1, 1, 4)
        self._init_plots()
        self._init_slider_vals()
        self.setLayout(layout)
    
    def _internal_connect(self):
        self._connect_sliders()
        #self.prev_next.connect(self.set_feature)
    
    def bounded_spec(self):
        feat_wv = self.feature.wv
        min_wv = feat_wv-1.5*self.display_width
        max_wv = feat_wv+1.5*self.display_width
        bspec = self.spectrum.bounded_sample((min_wv, max_wv))
        return bspec
    
    def sliders_changed(self, intval):
        #just ignore which slider caused the change get everything
        off = self.off_slider.value()
        gw = self.g_slider.value()
        lw = self.l_slider.value()
        depth = self.d_slider.value()
        self.feature.profile.set_parameters(np.array([off, gw, lw]))
        self.feature.set_depth(depth)
        self.update_plots()
        self.slidersChanged.emit(self.feature_idx)
    
    def set_feature(self, model_index):
        feature_idx = model_index.row()
        self.feature_idx = feature_idx
        self.feature = self.features[feature_idx]
        self.on_feature_changed()
    
    def on_feature_changed(self):
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
        d = self.feature.depth #always access depth before setting anything
        self.off_slider.set_value(off)
        self.g_slider.set_value(gw)
        self.l_slider.set_value(lw)
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
