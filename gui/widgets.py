import threading
import numpy as np
import matplotlib
#matplotlib.use('Qt4Agg')
#try: 
#from PySide.QtCore import *
#    from PySide.QtGui import *
#    matplotlib.rcParams['backend.qt4'] = 'PySide'
#except ImportError:
#    from PyQt4.QtCore import *
#    from PyQt4.QtGui import *
#    matplotlib.rcParams['backend.qt4'] = 'PyQt4'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

from models import *
from views import *
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
        try:
            if val > self.max:
                print "value above slider max, truncating"
                val = self.max
            elif val < self.min:
                print "value below slider min, truncating"
                val = self.min
            elif val == np.nan:
                print "value is nan, using minimum"
                val = self.min
            vfrac = (val-self.min)/self.delta
            opt_idx = int(np.around(vfrac*self.n_steps))
            self.slider.setValue(opt_idx)
        except:
            print "could not set slider value"

class MatplotlibCanvas (FigureCanvas):
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
    axes : the main axe object for the figure
    
    Methods
    -------
    
    Notes
    -----
    __1)__ adapted from S. Tosi, ``Matplotlib for Python Developers''
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
        self._axes = []
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
                        x_share_ax = self._axes[-col_idx]
                elif sharex == "columns":
                    if col_idx == 0:
                        x_share_ax = None
                    else:
                        x_share_ax = self._axes[-row_idx*ncols]
                elif sharex == "all":
                    x_share_ax = self._axes[0]
                else:
                    raise Exception("don't recognize sharex behavior")
                if sharey == "none":
                    y_share_ax = None
                elif sharey == "rows":
                    if col_idx == 0:
                        y_share_ax = None
                    else:
                        y_share_ax = self._axes[-col_idx]
                elif sharey == "columns":
                    if row_idx == 0:
                        y_share_ax = None
                    else:
                        y_share_ax = self._axes[-row_idx*ncols]
                elif sharey == "all":
                    y_share_ax = self._axes[0]
                else:
                    raise Exception("don't recognize sharey behavior")
                self._axes.append(self.fig.add_subplot(nrows, ncols, ax_num, sharex=x_share_ax, sharey=y_share_ax))
                ax_num += 1
        
        #make a shortcut for the first (and usually only) axis
        self.ax = self._axes[0]
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
        return self._axes[ax_num]
    
    def set_ax(row_idx, col_idx):
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
    def __init__(self, parent=None, nrows=1, ncols=1, mpl_toolbar=True,
                 sharex="none", sharey="none"):
        #self.parent = parent
        # initialization of Qt MainWindow widget
        QWidget.__init__(self, parent)
        
        # set the canvas to the Matplotlib widget
        self.canvas = MatplotlibCanvas(nrows, ncols, sharex=sharex, sharey=sharey)
        self.fig = self.canvas.fig
        
        # create a vertical box layout
        self.vboxlayout = QVBoxLayout()
        self.vboxlayout.addWidget(self.canvas)
        if mpl_toolbar:
            self.mpl_toolbar = NavigationToolbar(self.canvas,self)
            self.mpl_toolbar.setWindowTitle("Plot")
            self.vboxlayout.addWidget(self.mpl_toolbar)
        
        # set the layout to the vertical box
        self.setLayout(self.vboxlayout)
    
    @property
    def ax(self):
        return self.canvas.ax
    
    def set_ax(self, row, column):
        self.canvas.set_ax(row, column)
    
    def axis(self, row, column):
        return self.canvas.axis(row, column)
    
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
        self.prev_btn.clicked.connect(self.on_prev_clicked)
        self.next_btn.clicked.connect(self.on_next_clicked)
        self.play_toggle_btn.clicked.connect(self.toggle_pause)

    def on_prev_clicked(self):
        self.paused = True
        self.emit_prev()
    
    def on_next_clicked(self):
        self.paused=True
        self.emit_next()

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
    
    def __init__(self, spectrum, features, feature_idx, display_width, parent=None):
        super(FeatureFitWidget, self).__init__(parent)
        
        self.display_width = display_width
        self.spectrum = spectrum
        self.features = features
        self.feature = features[feature_idx]
        self.feature_idx = feature_idx
        
        self.lay = QGridLayout()
        
        self.mpl_fit = MatplotlibWidget(parent=parent, nrows=2, sharex="columns")
        self.lay.addWidget(self.mpl_fit, 1, 0, 2, 1)
        slider_orientation = Qt.Vertical
        self.off_slider = FloatSlider("offset", -0.15, 0.15, orientation=slider_orientation)
        self.d_slider = FloatSlider("depth", 0.0, 1.0, orientation=slider_orientation)
        self.g_slider = FloatSlider("sigma", 0.0, 1.0, orientation=slider_orientation)
        self.l_slider = FloatSlider("gamma", 0.0, 1.0, orientation=slider_orientation)
        slider_grid = [(2, 1, 1, 1), (2, 2, 1, 1), (2, 3, 1, 1), (2, 4, 1, 1)]
        slider_list = [self.off_slider, self.d_slider, self.g_slider, self.l_slider]
        for sl_idx in range(len(slider_list)):
            self.lay.addWidget(slider_list[sl_idx], *slider_grid[sl_idx])
        self.prev_next = PrevNext(duration=1.0, parent=self)
        self.lay.addWidget(self.prev_next, 1, 1, 1, 4)
        self._init_feature_table()
        self._init_plots()
        self._init_slider_vals()
        self._internal_connect()
        self.setLayout(self.lay)
    
    def fit_axis(self, row):
        return self.mpl_fit.axis(row, 0)
    
    def _internal_connect(self):
        self._connect_sliders()
        self.slidersChanged.connect(self.update_row)
        #print dir(self.linelist_view)
        #print ""
        #print self.linelist_view.selectionModel()
        #print dir(self.linelist_view.selectionModel())
        #self.linelist_view.selectionModel().currentRowChanged.connect(self.on_selection_change)
        self.prev_next.next.connect(self.next_feature)
        self.prev_next.prev.connect(self.prev_feature)
    
    def on_selection_change(self, row):
        print "in on selection change", row
        #print "in on_selection_change", selection
        #print dir(selection)
        
    def next_feature(self):
        #get our current index
        #print self.linelist_view.selectionModel().currentSelection()
        #self.linelist_view.setSelection()
        self.feature_idx = min(self.feature_idx + 1, self.linelist_model.rowCount()-1) 
        self.feature = self.features[self.feature_idx]
        self.on_feature_changed()
    
    def prev_feature(self):
        #self.linelist_view.currentSelection()
        self.feature_idx = max(self.feature_idx - 1, 0)
        self.feature = self.features[self.feature_idx]
        self.on_feature_changed()
    
    def _init_feature_table(self):
        drole = Qt.DisplayRole
        crole = Qt.CheckStateRole
        wvcol = Column("Wavelength", getter_dict = {drole: lambda x: "%10.3f" % x.wv})
        spcol = Column("Species", getter_dict = {drole: lambda x: "%10.3f" % x.species})
        epcol = Column("Excitation\nPotential", {drole: lambda x:"%10.3f" % x.ep})
        loggfcol = Column("log(gf)", {drole: lambda x: "%10.3f" % x.loggf})        
        offsetcol = Column("Offset", {drole: lambda x: "%10.3f" % x.get_offset()})
        depthcol = Column("Depth", {drole: lambda x: "%10.3f" % x.depth})
        ewcol = Column("Equivalent\nWidth", {drole: lambda x: "%10.3f" % x.eq_width})
        viewedcol = Column("Viewed", {crole: lambda x: x.flags["viewed"]}, checkable=True)
        #ewcol = Column("depth"
        columns = [wvcol, spcol, epcol, loggfcol, offsetcol, 
                   depthcol, ewcol, viewedcol]
        self.linelist_model = ConfigurableTableModel(self.features, columns)
        self.linelist_view = LineListView(parent=self)
        self.linelist_view.setModel(self.linelist_model)
        self.linelist_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.lay.addWidget(self.linelist_view, 0, 0, 1, 1)
    
    def update_row(self, row_num):
        left_idx = self.linelist_model.index(row_num, 0)
        right_idx = self.linelist_model.index(row_num, self.linelist_model.columnCount())
        self.linelist_model.dataChanged.emit(left_idx, right_idx)
    
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
        self.fit_axis(0).set_xlim(xlim_min, xlim_max)
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
        self.fit_axis(0).set_xlim(xlim_min, xlim_max)
        bspec = self.bounded_spec()
        self.data_line ,= self.fit_axis(0).plot(bspec.wv, bspec.flux, c="b")
        self.cont_line ,= self.fit_axis(0).plot(bspec.wv, bspec.norm, c="g")
        feature_model = self.feature.get_model_flux(bspec.wv)*bspec.norm
        self.model_line,= self.fit_axis(0).plot(bspec.wv, feature_model)
        nac = bspec.norm[len(bspec.norm)//2]
        self.top_marker_line ,= self.fit_axis(0).plot([feat_wv, feat_wv], [0.7*nac, 1.1*nac], c="r", lw=1.5) 
        
        self.bottom_marker_line ,= self.fit_axis(1).plot([feat_wv, feat_wv], [-10.0, 10.0], c="r", lw=1.5) 
        #import pdb; pdb.set_trace()
        #and now for the residuals plot
        inv_var = bspec.get_inv_var()
        bkground_alpha = 0.5
        self.zero_line ,= self.fit_axis(1).plot([bspec.wv[0], bspec.wv[-1]], [0, 0], c="k", alpha=bkground_alpha, lw=2.0)
        sig_levels = [3]
        self.sig_lines = [self.fit_axis(1).plot([bspec.wv[0], bspec.wv[-1]], [sl, sl], c="k", alpha=bkground_alpha)[0] for sl in sig_levels]
        self.sig_lines.extend([self.fit_axis(1).plot([bspec.wv[0], bspec.wv[-1]], [-sl, -sl], c="k", alpha=bkground_alpha)[0] for sl in sig_levels])
        
        #plot the model residuals. 
        significance = np.sqrt(inv_var)*(feature_model-bspec.flux)
        self.resid_line ,= self.fit_axis(1).plot(bspec.wv, significance, c="b")
        self.fit_axis(1).set_ylim(-6, 6)
        self.mpl_fit.draw()
    
    def update_plots(self):
        feat_wv = self.feature.wv
        bspec = self.bounded_spec()
        self.data_line.set_data(bspec.wv, bspec.flux)
        bnorm = bspec.norm
        self.cont_line.set_data(bspec.wv, bnorm)
        feature_model = self.feature.get_model_flux(bspec.wv)*bnorm
        self.model_line.set_data(bspec.wv, feature_model)
        nac = bspec.norm[len(bspec.norm)//2]
        self.top_marker_line.set_data([feat_wv, feat_wv], [0.7*nac, 1.1*nac])
        self.bottom_marker_line.set_xdata([feat_wv, feat_wv])

        inv_var = bspec.get_inv_var()
        significance = (feature_model-bspec.flux)*np.sqrt(inv_var)
        self.resid_line.set_data(bspec.wv, significance)
        self.zero_line.set_data([bspec.wv[0], bspec.wv[-1]], [0, 0])
        for line in self.sig_lines:
            line.set_xdata([bspec.wv[0], bspec.wv[-1]])
        
        self.mpl_fit.draw()
