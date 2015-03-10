import threading
import numpy as np
import matplotlib

from PySide import QtCore
from PySide import QtGui
from PySide.QtCore import Signal, Slot
Qt = QtCore.Qt

class FeatureFitWidget(QtGui.QWidget):
    slidersChanged = Signal(int)
    
    def __init__(self, features, feature_idx, parent=None):
        super(FeatureFitWidget, self).__init__(parent)
        self.display_width = options.display_width
        #self.spectra = spectra
        self.features = features
        self.feature = features[feature_idx]
        self.feature_idx = feature_idx
        self.norm_hint_wvs = []
        self.norm_hint_fluxes = []
        
        self.lay = QtGui.QGridLayout()
        
        self.mpl_fit = MatplotlibWidget(parent=parent, nrows=2, sharex="columns")
        self.lay.addWidget(self.mpl_fit, 1, 0, 3, 1)
        slider_orientation = Qt.Vertical
        slider_n_steps = 200
        self.off_slider = FloatSlider("offset", -0.15, 0.15, orientation=slider_orientation, n_steps=slider_n_steps)
        self.d_slider = FloatSlider("depth", 0.0, 1.0, orientation=slider_orientation, n_steps=slider_n_steps)
        self.g_slider = FloatSlider("sigma", 0.0, 1.0, orientation=slider_orientation, n_steps=slider_n_steps)
        self.l_slider = FloatSlider("gamma", 0.0, 1.0, orientation=slider_orientation, n_steps=slider_n_steps)
        self.cont_slider = FloatSlider("rel norm", 0.90, 1.10, orientation=slider_orientation, n_steps=slider_n_steps)
        slider_grid = [(2, 1, 1, 1), (2, 2, 1, 1), (2, 3, 1, 1), (2, 4, 1, 1), (2, 5, 1, 1)]
        slider_list = [self.off_slider, self.d_slider, self.g_slider, self.l_slider, self.cont_slider]
        for sl_idx in range(len(slider_list)):
            self.lay.addWidget(slider_list[sl_idx], *slider_grid[sl_idx])
        
        #previous/next setup
        self.prev_next = PrevNext(duration=1.0, parent=self)
        self.lay.addWidget(self.prev_next, 1, 1, 1, 4)
        
        #output_file button
        self.output_button = QtGui.QPushButton("save measurements")
        self.output_button.clicked.connect(self.save_measurements)
        self.lay.addWidget(self.output_button, 3, 1, 1, 2)
        
        #use check box
        self.use_cb = QtGui.QCheckBox("Use line")
        self.use_cb.setChecked(self.feature.flags["use"])
        self.lay.addWidget(self.use_cb, 3, 3, 1, 1)
        
        self._init_feature_table()
        self._init_plots()
        self._init_slider_vals()
        self._internal_connect()
        self.setLayout(self.lay)

    def minimumSizeHint(self):
        return QtGui.QSize(500, 500)
    
    def save_feature_fits(self, fname):
        import cPickle
        cPickle.dump(self.features, open(fname, "wb"))
    
    def save_measurements(self):
        fname, file_filter = QtGui.QFileDialog.getSaveFileName(self, "save measurements")
        try:
            tmb.io.linelist_io.write_moog_from_features(fname, self.features)
        except Exception as e:
            print e
        try:
            feat_fname = ".".join(fname.split(".")[:]) + ".features.pkl"
            self.save_feature_fits(feat_fname)
        except Exception as e:
            print e
    
    @property
    def hint_click_on(self):
        return False
    
    def handle_plot_click(self, eventl):
        event ,= eventl
        #print "clicked!", event.button
        if event.button == 2:
            if self.hint_click_on:
                hwv = event.xdata
                hflux = event.ydata
                self.add_norm_hint(hwv, hflux)
    
    def add_norm_hint(self, wv, flux):
        self.norm_hint_wvs.append(wv)
        self.norm_hint_fluxes.append(flux) 
        #todo add a realistic error estimate for the hints
        hint_tuple = self.norm_hint_wvs, self.norm_hint_fluxes, np.ones(len(self.norm_hint_wvs), dtype=float)*10.0
        tmb.utils.misc.approximate_normalization(self.spectrum,norm_hints=hint_tuple,overwrite=True)
        self.update_plots()
    
    def fit_axis(self, row):
        return self.mpl_fit.axis(row, 0)
    
    def _internal_connect(self):
        self.mpl_fit.buttonPressed.connect(self.handle_plot_click)
        
        self._connect_sliders()
        self.slidersChanged.connect(self.update_row)
        #print self.linelist_view.selectionModel()
        #print dir(self.linelist_view.selectionModel())
        #self.linelist_view.selectionModel().currentRowChanged.connect(self.on_selection_change)
        self.linelist_view.doubleClicked.connect(self.set_feature)
        self.prev_next.next.connect(self.next_feature)
        self.prev_next.prev.connect(self.prev_feature)
        self.use_cb.stateChanged.connect(self.set_use)
    
    def set_use(self, state_val):
        self.feature.flags["use"] = state_val > 0
    
    def on_selection_change(self, row):
        print "in on selection change", row
        #print "in on_selection_change", selection
        #print dir(selection)
    
    def set_feature(self, index):
        row = index.row()
        self.feature_idx = row
        self.feature = self.features[self.feature_idx]
        self.linelist_view.selectRow(self.feature_idx)
        self.on_feature_changed()
    
    def next_feature(self):
        next_idx = self.feature_idx + 1
        if next_idx > self.linelist_model.rowCount()-1:
            next_idx = self.linelist_model.rowCount()-1
            self.prev_next.pause() 
        self.feature_idx = next_idx
        self.feature = self.features[self.feature_idx]
        self.linelist_view.selectRow(self.feature_idx)
        self.on_feature_changed()
    
    def prev_feature(self):
        prev_idx = self.feature_idx - 1
        if prev_idx < 0:
            prev_idx = 0
            self.prev_next.pause() 
        self.feature_idx = prev_idx
        self.feature_idx = max(self.feature_idx - 1, 0)
        self.feature = self.features[self.feature_idx]
        self.linelist_view.selectRow(self.feature_idx)
        self.on_feature_changed()
    
    def _init_feature_table(self):
        drole = Qt.DisplayRole
        crole = Qt.CheckStateRole
        wvcol = models.Column("Wavelength", getter_dict = {drole: lambda x: "%10.3f" % x.wv})
        spcol = models.Column("Species", getter_dict = {drole: lambda x: "%10.3f" % x.species})
        epcol = models.Column("Excitation\nPotential", {drole: lambda x:"%10.3f" % x.ep})
        loggfcol = models.Column("log(gf)", {drole: lambda x: "%10.3f" % x.loggf})        
        offsetcol = models.Column("Offset", {drole: lambda x: "%10.3f" % x.get_offset()})
        depthcol = models.Column("Depth", {drole: lambda x: "%10.3f" % x.depth})
        sigcol = models.Column("sigma", {drole: lambda x: "% 10.3f" % x.profile.get_parameters()[1]})
        gamcol = models.Column("gamma", {drole: lambda x: "% 10.3f" % x.profile.get_parameters()[2]})
        ewcol = models.Column("Equivalent\nWidth", {drole: lambda x: "%10.2f" % (1000.0*x.eq_width)})
        def set_note(x, note):
            x.note = note
            return True
        notescol = models.Column("Notes", {drole:lambda x: x.note}, setter_dict={Qt.EditRole: set_note}, editable=True)
        #viewedcol = Column("Viewed", getter_dict={crole: dummy_func}, setter_dict={crole: flag_setter_factory("viewed")}, checkable=True)
        
        #ewcol = Column("depth"
        columns = [wvcol, spcol, epcol, loggfcol, offsetcol, 
                   depthcol, sigcol, gamcol, ewcol, notescol]#, viewedcol]
        self.linelist_model = models.ConfigurableTableModel(self.features, columns)
        self.linelist_view = views.LineListView(parent=self)
        self.linelist_view.setModel(self.linelist_model)
        self.linelist_view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.lay.addWidget(self.linelist_view, 0, 0, 1, 6)
    
    def update_row(self, row_num):
        left_idx = self.linelist_model.index(row_num, 0)
        right_idx = self.linelist_model.index(row_num, self.linelist_model.columnCount())
        self.linelist_model.dataChanged.emit(left_idx, right_idx)
    
    def bounded_spec(self):
        feat_wv = self.feature.wv
        #min_wv = feat_wv-1.5*self.display_width
        #max_wv = feat_wv+1.5*self.display_width
        bspec = self.feature.data_sample
        return bspec
    
    def sliders_changed(self, intval):
        #just ignore which slider caused the change get everything
        off = self.off_slider.value()
        gw = self.g_slider.value()
        lw = self.l_slider.value()
        depth = self.d_slider.value()
        relc = self.cont_slider.value()
        self.feature.profile.set_parameters(np.asarray([off, gw, lw]))
        self.feature.set_relative_continuum(relc)
        self.feature.set_depth(depth)
        self.update_plots()
        self.slidersChanged.emit(self.feature_idx)
    
    def on_feature_changed(self):
        if self.feature.flags["use"]:
            self.use_cb.setChecked(True)
        else:
            self.use_cb.setChecked(False)
        self._init_slider_vals()
        feat_wv = self.feature.wv
        xlim_min = feat_wv-self.display_width
        xlim_max = feat_wv+self.display_width
        self.fit_axis(0).set_xlim(xlim_min, xlim_max)
        bspec = self.bounded_spec()
        ymin, ymax = np.min(bspec.flux), np.max(bspec.flux)
        ydelta = ymax-ymin
        extra_frac = 0.05
        self.fit_axis(0).set_ylim(ymin-extra_frac*ydelta, ymax+extra_frac*ydelta)
        self.update_plots()
    
    def _connect_sliders(self):
        self.off_slider.slider.valueChanged.connect(self.sliders_changed)
        self.g_slider.slider.valueChanged.connect(self.sliders_changed)
        self.l_slider.slider.valueChanged.connect(self.sliders_changed)
        self.d_slider.slider.valueChanged.connect(self.sliders_changed)
        self.cont_slider.slider.valueChanged.connect(self.sliders_changed)
    
    def _init_slider_vals(self):
        off, gw, lw = self.feature.profile.get_parameters()
        d = self.feature.depth #always access depth before setting anything
        relc = self.feature.relative_continuum
        self.off_slider.set_value(off)
        self.g_slider.set_value(gw)
        self.l_slider.set_value(lw)
        self.d_slider.set_value(d)
        self.cont_slider.set_value(relc)
    
    def _init_plots(self):
        feat_wv = self.feature.wv
        xlim_min = feat_wv-self.display_width
        xlim_max = feat_wv+self.display_width
        self.fit_axis(0).set_xlim(xlim_min, xlim_max)
        bspec = self.bounded_spec()
        self.data_line ,= self.fit_axis(0).plot(bspec.wv, bspec.flux, c="b")
        self.cont_line ,= self.fit_axis(0).plot(bspec.wv, bspec.norm, c="g")
        feature_model = self.feature.model_flux(bspec.wv)*bspec.norm
        self.model_line,= self.fit_axis(0).plot(bspec.wv, feature_model)
        nac = bspec.norm[len(bspec.norm)//2]
        self.top_marker_line ,= self.fit_axis(0).plot([feat_wv, feat_wv], [0.7*nac, 1.1*nac], c="r", lw=1.5) 
        
        self.bottom_marker_line ,= self.fit_axis(1).plot([feat_wv, feat_wv], [-10.0, 10.0], c="r", lw=1.5) 
        #import pdb; pdb.set_trace()
        #and now for the residuals plot
        inv_var = bspec.ivar
        bkground_alpha = 0.5
        self.zero_line ,= self.fit_axis(1).plot([bspec.wv[0], bspec.wv[-1]], [0, 0], c="k", alpha=bkground_alpha, lw=2.0)
        sig_levels = [3]
        self.sig_lines = [self.fit_axis(1).plot([bspec.wv[0], bspec.wv[-1]], [sl, sl], c="k", alpha=bkground_alpha)[0] for sl in sig_levels]
        self.sig_lines.extend([self.fit_axis(1).plot([bspec.wv[0], bspec.wv[-1]], [-sl, -sl], c="k", alpha=bkground_alpha)[0] for sl in sig_levels])
        
        #plot the model residuals. 
        significance = np.sqrt(inv_var)*(feature_model-bspec.flux)
        self.resid_line ,= self.fit_axis(1).plot(bspec.wv, significance, c="b")
        self.fit_axis(1).set_ylim(-6, 6)
        self.fit_axis(1).set_xlabel("Wavelength")
        self.fit_axis(1).set_ylabel("Residual Significance")
        self.fit_axis(0).set_ylabel("Flux")
        self.mpl_fit.draw()
    
    def update_plots(self):
        feat_wv = self.feature.wv
        bspec = self.bounded_spec()
        self.data_line.set_data(bspec.wv, bspec.flux)
        bnorm = bspec.norm
        self.cont_line.set_data(bspec.wv, bnorm)
        feature_model = self.feature.model_flux(bspec.wv)*bnorm
        self.model_line.set_data(bspec.wv, feature_model)
        nac = bspec.norm[len(bspec.norm)//2]
        self.top_marker_line.set_data([feat_wv, feat_wv], [0.7*nac, 1.1*nac])
        self.bottom_marker_line.set_xdata([feat_wv, feat_wv])
        
        inv_var = bspec.ivar
        significance = (feature_model-bspec.flux)*np.sqrt(inv_var)
        self.resid_line.set_data(bspec.wv, significance)
        self.zero_line.set_data([bspec.wv[0], bspec.wv[-1]], [0, 0])
        for line in self.sig_lines:
            line.set_xdata([bspec.wv[0], bspec.wv[-1]])
        
        self.mpl_fit.draw()
