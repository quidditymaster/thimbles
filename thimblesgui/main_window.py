
# standard library
from itertools import cycle, product
import os
import sys
import time
import cPickle

# 3rd party packages
import matplotlib
matplotlib.use('Qt4Agg')
from PySide.QtCore import *
from PySide.QtGui import *
matplotlib.rcParams['backend.qt4'] = 'PySide'
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize

import thimblesgui as tmbg
import thimbles as tmb
import thimbles.io as io

_resources_dir = os.path.join(os.path.dirname(__file__),"resources")

# ########################################################################### #

class AppForm(QMainWindow):
    
    def __init__(self, options, rows=[]):
        super(AppForm, self).__init__()
        self.setWindowTitle("Thimbles")
        self.main_frame = QWidget()        
        self.options = options
        self.layout = QHBoxLayout()
        self.main_table_model = tmbg.models.MainTableModel()
        self.rfunc = tmbg.user_namespace.eval_("tmb.io."+options.read_func)
        
        self.load_linelist()
        
        
        if self.options.batch_mode:
            input_files = open(options.files[0], "r").readlines()
            input_files = [fname.rstrip() for fname in input_files]  
        else:
            input_files = options.files
        
        print "input files", input_files
        
        for sfile_name in input_files:
            try:
                joined_name = os.path.join(options.data_dir, sfile_name)
                spec_list = self.rfunc(joined_name)
            except Exception as e:
                print "there was an error reading file %s" % sfile_name
                print e
            spec_base_name = os.path.basename(sfile_name)
            fit_features = self.pre_process_spectra(spec_list)
            spec_base_name = os.path.basename(sfile_name)
            features_name = "features_%s" % spec_base_name
            if not self.options.no_window:
                spec_row = tmbg.models.SpectraRow(spec_list, spec_base_name)
                self.main_table_model.addRow(spec_row)
                if not fit_features is None:
                    frow = tmbg.models.FeaturesRow(fit_features, features_name)
                    self.main_table_model.addRow(frow)
            
            if self.options.features_out:
                self.save_features(spec_base_name, fit_features)
            
            if self.options.moog_out:
                self.save_moog_from_features(spec_base_name, fit_features)
        
        #setup for the dual spectrum operations
        self.partial_result = None
        self.current_operation = None

        self.main_table_view = tmbg.views.NameTypeTableView(self)
        self.main_table_view.setModel(self.main_table_model)
        self.main_table_view.setColumnWidth(0, 200)
        self.main_table_view.setColumnWidth(1, 200)
        self.main_table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.main_table_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.layout.addWidget(self.main_table_view)
        
        #import pdb; pdb.set_trace()
                
        op_gb = self._init_operations_groups()
        self.layout.addWidget(op_gb)
        
        self.main_frame.setLayout(self.layout)
        self.setCentralWidget(self.main_frame)
        
        #self.create_menu()
        #self._init_actions()
        #self._init_menus()
        
        self._init_status_bar()
        self._connect()
    
    def print_args(self, *args, **kwargs):
        print "in print_args"
        print args, kwargs
    
    def _init_operations_groups(self):
        all_op_box = QGroupBox("spectral operations")
        top_layout = QVBoxLayout()
        
        mono_box = self._init_mono_operations()
        dual_box = self._init_dual_operations()
        multi_box = self._init_multi_operations()
        
        top_layout.addWidget(mono_box)
        top_layout.addWidget(dual_box)
        top_layout.addWidget(multi_box)
        all_op_box.setLayout(top_layout)
        return all_op_box
    
    def _init_mono_operations(self):
        op_box = QGroupBox("mono spectrum operations")
        btn_grid = QGridLayout()
        self.load_btn = QPushButton("load")
        self.norm_btn = QPushButton("norm")
        self.rv_btn = QPushButton("set rv")
        self.extract_order_btn = QPushButton("extract order")
        self.ex_norm_btn = QPushButton("extract normalized")
        self.fit_features_btn = QPushButton("fit features")
        #self.tell_btn = QPushButton("extract telluric")
        btn_grid.addWidget(self.load_btn, 0, 0, 1, 1)
        btn_grid.addWidget(self.norm_btn, 1, 0, 1, 1)
        btn_grid.addWidget(self.rv_btn, 2, 0, 1, 1)
        btn_grid.addWidget(self.extract_order_btn, 3, 0, 1, 1)
        btn_grid.addWidget(self.fit_features_btn, 4, 0, 1, 1)
        #btn_grid.addWidget(self.tell_btn, 1, 0, 1, 1)
        op_box.setLayout(btn_grid)
        return op_box
    
    def _init_dual_operations(self):
        op_box = QGroupBox("paired spectrum operations")
        btn_grid = QGridLayout()
        self.add_btn = QPushButton("+")
        self.sub_btn = QPushButton("-")
        self.mul_btn = QPushButton("*")
        self.div_btn = QPushButton("/")
        btn_grid.addWidget(self.add_btn, 0, 0, 1, 1)
        btn_grid.addWidget(self.sub_btn, 0, 1, 1, 1)
        btn_grid.addWidget(self.mul_btn, 1, 0, 1, 1)
        btn_grid.addWidget(self.div_btn, 1, 1, 1, 1)
        op_box.setLayout(btn_grid)
        return op_box
    
    def _init_multi_operations(self):
        op_box = QGroupBox("multi spectrum operations")
        btn_grid = QGridLayout()
        self.coadd_btn = QPushButton("coadd")
        self.compare_btn =QPushButton("compare")
        btn_grid.addWidget(self.coadd_btn, 0, 0, 1, 1)
        btn_grid.addWidget(self.compare_btn, 1, 0, 1, 1)
        op_box.setLayout(btn_grid)
        return op_box
    
    def _connect(self):
        self.main_table_view.doubleClicked.connect(self.on_double_click)
        self.rv_btn.clicked.connect(self.on_set_rv)
        self.div_btn.clicked.connect(self.on_div)
        self.mul_btn.clicked.connect(self.on_mul)
        self.add_btn.clicked.connect(self.on_add)
        self.sub_btn.clicked.connect(self.on_sub)
        self.load_btn.clicked.connect(self.on_load)
        self.fit_features_btn.clicked.connect(self.on_fit_features)
        self.norm_btn.clicked.connect(self.on_norm)
        self.compare_btn.clicked.connect(self.on_compare)
        self.extract_order_btn.clicked.connect(self.on_extract_order)
    
    def bad_selection(self, msg=None):
        """indicate when operations cannot be performed because of bad user selections
        """
        if msg == None:
            msg = "invalid selection\n"
        else:
            msg = "invalid selection\n" + msg
        self.wd = tmbg.dialogs.WarningDialog(msg)
        self.wd.warn()
    
    def get_row(self, row):
        return self.main_table_model.rows[row]
    
    def match_standard(self, spectra):
        return 
    
    def on_double_click(self, index):
        col_index = index.column()
        if col_index == 0:
            return
        row_index = index.row()
        row_object = self.get_row(row_index)
        row_object.on_double_click()
    
    def on_compare(self):
        smod = self.main_table_view.selectionModel()
        selrows = smod.selectedRows()
        row_objs = [self.get_row(r.row()) for r in selrows]
        row_types = [r.type_id for r in row_objs]
        row_data = [r.data for r in row_objs]
        type_ids = list(set(row_types))
        if len(type_ids) == 1:
            cur_type ,= type_ids
            color_cycle = cycle("bgrky")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if cur_type == "spectra":
                for spec_set in row_data:
                    cur_c = color_cycle.next()
                    for spec in spec_set:
                        ax.plot(spec.wv, spec.flux/spec.norm, c=cur_c)
            plt.show()
        elif len(type_ids) == 0:
            self.bad_selection("nothing selected to compare!")
        else:
            self.bad_selection("not all types matched")
    
    def on_delete(self):
        smod = self.main_table_view.selectionModel()
        selrows = smod.selectedRows()
        self.main_table_model.beginRemoveRows()
    
    def on_add(self):
        self.on_op("+")
    
    def on_sub(self):
        self.on_op("-")
    
    def on_div(self):
        self.on_op("/")
    
    def on_mul(self):
        self.on_op("*")
    
    def on_op(self, operation):
        smod = self.main_table_view.selectionModel()
        selrows = smod.selectedRows()
        if len(selrows) != 2:
            self.bad_selection("need 2 spectra selected")
            return
        row_idx1, row_idx2 = selrows[0].row(), selrows[1].row()
        row1, row2 = self.get_row(row_idx1), self.get_row(row_idx2)
        if row1.type_id != "spectra" or row2.type_id != "spectra":
            self.bad_selection("operations only work on 2 spectra")
            return
        if len(row1.data) != len(row2.data):
            self.wd = tmbg.dialogs.WarningDialog("incompatible numbers of spectral orders")
            self.wd.warn()
            return
        
    
    def on_extract_order(self):
        smod = self.main_table_view.selectionModel()
        selrows = smod.selectedRows()
        if len(selrows) > 1:
            self.bad_selection("extract one at a time")
            return
        row_idx = selrows[0].row()
        row = self.get_row(row_idx)
        if row.type_id != "spectra":
            self.bad_selection("can only extract orders from spectra")
            return
        ex_ord_res = QInputDialog.getInt(self, "extract order dialog", "enter order number (0 indexed)")
        ex_ord, input_success = ex_ord_res
        if not input_success:
            return
        new_name = "%s_order_%d" % (row.name, ex_ord)
        
        new_spec = [row.data[ex_ord]]
        self.main_table_model.addRow(tmbg.models.SpectraRow(new_spec, new_name))
    
    def on_fit_features(self):
        smod = self.main_table_view.selectionModel()
        selrows = smod.selectedRows()
        if len(selrows) != 2:
            self.bad_selection("need one line list and one spectrum selected")
            return
        row1, row2 = selrows[0].row(), selrows[1].row()
        spec = None
        ll = None
        for row_index in [row1, row2]:
            row = self.get_row(row_index)
            if row.type_id == "spectra":
                spec = row.data
                spec_name = row.name
            elif row.type_id == "line list":
                ll = row.data
                ll_name = row.name
        if spec != None and ll != None:
            culled_features = self.pre_cull_lines(spec, ll)
            if len(culled_features) == 0:
                self.wd = tmbg.dialogs.WarningDialog("There were no features in the overlap! \n Check your wavelength solution")
                self.wd.warn()
                return
            fit_features = self.fit_features(culled_features)
            features_name = "features from %s %s" % (spec_name, ll_name)
            frow = tmbg.models.FeaturesRow(fit_features, features_name)
            self.main_table_model.addRow(frow)
        else:
            self.bad_selection("need one line list and one spectrum selected")
    
    def on_set_rv(self):
        smod = self.main_table_view.selectionModel()
        selrows = smod.selectedRows()
        if len(selrows) != 1:
            return
        row = self.get_row(selrows[0].row())
        if row.type_id == "spectra":
            rvdialog = tmbg.dialogs.RVSettingDialog(row.data, self)
            row.widgets["rv"] = rvdialog
            rvdialog.set_rv()
    
    def on_norm(self):
        smod = self.main_table_view.selectionModel()
        selrows = smod.selectedRows()
        row_idxs = [r.row() for r in selrows]
        row_objs = [self.get_row(idx) for idx in row_idxs]
        for row in row_objs:
            if row.type_id == "spectra":
                nd = tmbg.dialogs.NormalizationDialog(row.data)
                nd.get_norm()
                #for spec in row.data:
                #    spec.approx_norm()
    
    def on_load(self):
        ld = tmbg.dialogs.LoadDialog()
        new_row = ld.get_row()
        if isinstance(new_row, tmbg.models.MainTableRow):
            self.main_table_model.addRow(new_row)
    
    def pre_cull_lines(self, spectra, ldat):
        accepted_mask = np.zeros(len(ldat), dtype=bool)
        line_spec_idxs = np.zeros(len(ldat), dtype=int)
        for spec_idx in range(len(spectra)):
            spec = spectra[spec_idx]
            min_wv = np.min(spec.wv)
            max_wv = np.max(spec.wv)
            for feat_idx in range(len(ldat)):
                cwv, cid, cep, cloggf = ldat[feat_idx]
                samp_bounds = (cwv-self.options.fit_width, cwv+self.options.fit_width)
                bspec = spectra[spec_idx].bounded_sample(samp_bounds)
                if not bspec is None:
                    if len(bspec) > 3:
                        accepted = False
                        if self.options.pre_cull=="snr":
                            min_snr = np.min(bspec.flux*np.sqrt(bspec.get_inv_var()))
                            if min_snr > 10:
                                accepted=True
                        else:
                            accepted = True
                            print "no pre culling of linelist done"
                        if accepted:
                            accepted_mask[feat_idx] = True
                            line_spec_idxs[feat_idx] = spec_idx
        culled_features = []
        for feat_idx in range(len(ldat)):
            if accepted_mask[feat_idx]:
                sample_width = max(self.options.display_width, self.options.fit_width)
                cwv, cid, cep, cloggf = ldat[feat_idx]
                sample_bounds = (cwv-sample_width, cwv+sample_width)
                bspec = spectra[line_spec_idxs[feat_idx]].bounded_sample(sample_bounds, copy=False)
                
                tp = tmb.features.AtomicTransition(cwv, cid, cloggf, cep)
                wvdel = np.abs(bspec.wv[1]-bspec.wv[0])
                start_p = np.array([0.0, wvdel, 0.0])
                lprof = tmb.line_profiles.Voigt(cwv, start_p)
                nf = tmb.features.Feature(lprof, 0.00, 0.00, tp, data_sample=bspec)
                culled_features.append(nf)
        return culled_features
    
    def fit_features(self, features):
        self.quick_fit(features)
        for i in range(int(self.options.iteration)):
            self.preconditioned_feature_fit(features)
        return features
    
    def quick_fit(self, features):
        for feature in features:
            bspec = feature.data_sample
            wvs = bspec.wv
            cent_wv = feature.wv
            flux = bspec.flux
            minima = tmb.utils.misc.get_local_maxima(-flux)
            if np.sum(minima) == 0:
                feature.set_eq_width(0.0)
                continue
            minima_idxs = np.where(minima)[0]
            minima_wvs = wvs[minima_idxs]
            best_minimum_idx = np.argmin(np.abs(minima_wvs-cent_wv))
            closest_idx = minima_idxs[best_minimum_idx]
            fit_center, fit_sigma, fit_y = tmb.utils.misc.local_gaussian_fit(yvalues=flux, peak_idx=closest_idx, fit_width=2, xvalues=wvs)
            norm_flux = bspec.norm[closest_idx]
            depth = (norm_flux - fit_y)/norm_flux
            offset = fit_center-cent_wv
            new_params = [offset, np.abs(fit_sigma), 0.0]
            feature.profile.set_parameters(new_params)
            feature.set_depth(depth)
        return features
        
    
    def load_linelist(self):
        ldat = None
        if not self.options.line_list is None:
            try:
                ldat = np.loadtxt(self.options.line_list ,skiprows=1, usecols=[0, 1, 2, 3])
                ll_base_name = os.path.basename(self.options.line_list)
                ll_row = tmbg.models.LineListRow(ldat, ll_base_name)
                self.main_table_model.addRow(ll_row)
            except Exception as e:
                print "there was an error reading file %s" % self.options.line_list
                print e
        self.ldat = ldat
    
    def save_features(self, fname, features):
        out_fname=fname.split(".")[0] + ".features.pkl"
        out_fpath = os.path.join(self.options.output_dir, out_fname)
        cPickle.dump(features, open(out_fpath, "w"))
    
    def save_moog_from_features(self, fname, features):
        out_fname=fname.split(".")[0] + ".features.ln"
        out_fpath = os.path.join(self.options.output_dir, out_fname)
        io.linelist_io.write_moog_from_features(out_fpath, features)
    
    def pre_process_spectra(self, spectra):
        #apply the normalization
        if self.options.norm == "auto":
            for spec in spectra:
                spec.approx_norm()
        
        #apply the radial velocity shift
        import pdb; pdb.set_trace()
        if self.options.rv == "cc":
            import h5py
            hf = h5py.File("/home/tim/data/caelho_grid/extracted_log_linear.h5")
            best_template = tmb.Spectrum(np.array(hf["wv"]), np.array(hf["flux"]))
            #best_template = self.match_standard(spectra)
            import pdb; pdb.set_trace()
            rv = tmb.velocity.template_rv_estimate(spectra, template=best_template, delta_max=self.options.max_rv)
        else:
            rv = float(self.options.rv)
        for spec in spectra:
            spec.set_rv(rv)
        
        if self.ldat is None:
            print "cannot carry out a fit without a feature line list"
            return None
        
        if self.options.fit == "individual":
            culled_features = self.pre_cull_lines(spectra, self.ldat)
            fit_features = self.fit_features(culled_features)
        return fit_features
    
    def preconditioned_feature_fit(self, features):
        lparams = np.array([f.profile.get_parameters() for f in features])
        sig_med = np.median(lparams[:, 1])
        sig_mad = np.median(np.abs(lparams[:, 1]-sig_med))
        
        print "sig_med", sig_med, "sig_mad", sig_mad
        
        cent_wvs = np.array([f.wv for f in features])
        vel_offs = lparams[:, 0]/cent_wvs
        
        vel_med = np.median(vel_offs)
        vel_mad = np.median(np.abs(vel_offs - vel_med))
        print "vel median", vel_med, "vel mad", vel_mad
        
        gam_med = np.median(np.abs(lparams[:, 2]))
        gam_mad = np.median(np.abs(lparams[:, 2]-gam_med))
        print "gam med", gam_med, "gam_mad", gam_mad
        
        gam_thresh = self.options.gamma_max
        
        def resids(pvec, wvs, lprof, nflux):
            pr = lprof.get_profile(wvs, pvec[2:])
            ew, relnorm, _off, g_sig, l_sig = pvec
            g_sig = np.abs(g_sig)
            l_sig = np.abs(l_sig)
            sig_reg = 0
            rndiff = np.abs(relnorm-1.0)
            sig_reg += 5.0*np.abs(rndiff)
            wv_delta = np.abs(wvs[-1]-wvs[0])/len(wvs)
            vel_off = _off/wvs[int(len(wvs)/2)]
            vel_diff =  np.abs(vel_off-vel_med)
            if vel_diff > 3.0*vel_mad:
                sig_reg += (vel_diff - 3.0*vel_mad)/vel_mad
            sig_diff = np.abs(g_sig-sig_med)
            if sig_diff > 3.0*sig_mad:
                sig_reg += (sig_diff-3.0*sig_mad)/sig_mad
            if np.abs(l_sig) > gam_thresh:
                sig_reg += np.abs((l_sig-gam_thresh))/np.max(0.1*wv_delta, gam_mad)
            gam_diff = np.abs(l_sig-gam_med)
            sig_reg += gam_diff/np.max(0.3*wv_delta, gam_mad)
            fdiff = nflux-(1.0-ew*pr)*relnorm
            return np.hstack((fdiff ,sig_reg))
        
        for feat_idx in range(len(features)):
            feat = features[feat_idx]
            cwv = feat.wv
            delta_wv = self.options.fit_width
            fit_bounds = (cwv-delta_wv, cwv+delta_wv)
            bspec = feat.data_sample.bounded_sample(fit_bounds)
            start_p = feat.profile.get_parameters()
            start_p[1:] = np.abs(start_p[1:])
            if np.abs(start_p[1]-sig_med) > 3.0*sig_mad:
                start_p[1] = sig_med
            if np.abs(start_p[2]-gam_med) > 3.0*gam_mad:
                start_p[2] = gam_med
            guessv = np.hstack((feat.eq_width, 1.0, start_p))
            nflux = bspec.flux/bspec.norm
            lprof=feat.profile
            fit_res = scipy.optimize.leastsq(resids, guessv, args=(bspec.wv, lprof, nflux))
            fit = fit_res[0]
            fit[3:] = np.abs(fit[3:])
            lprof.set_parameters(fit[2:])
            feat.relative_continuum = fit[1]
            feat.set_eq_width(fit[0]) 
        return features
    
    def _init_fit_widget(self):
        self.fit_widget = tmbg.widgets.FeatureFitWidget(self.spec, self.features, 0, self.options.fwidth, parent=self)
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 1)
    
    def save (self):
        QMessageBox.about(self, "Save MSG", "SAVE THE DATA\nTODO")
    
    def undo (self):
        QMessageBox.about(self, "Undo", "UNDO THE DATA\nTODO")
    
    def redo (self):
        QMessageBox.about(self, "Redo", "REDO THE DATA\nTODO")

    def _init_actions(self):
        
        self.menu_actions = {}
        
        self.menu_actions['save'] = QAction(QIcon(_resources_dir+'/images/save.png'),
                "&Save...", self, shortcut=QKeySequence.Save,
                statusTip="Save the current data",
                triggered=self.save)
        
        self.menu_actions['save as'] = QAction(QIcon(_resources_dir+'/images/save_as.png'),
                "&Save As...", self, shortcut=QKeySequence.SaveAs,
                statusTip="Save the current data as....",
                triggered=self.save)
        
        self.menu_actions['undo'] = QAction(QIcon(_resources_dir+'/images/undo_24.png'),
                "&Undo", self, shortcut=QKeySequence.Undo,
                statusTip="Undo the last editing action", triggered=self.undo)
        
        self.menu_actions['redo'] =  QAction(QIcon(_resources_dir+'/images/redo_24.png'),
                                                   "&Redo", self, shortcut=QKeySequence.Redo,
                                                   statusTip="Redo the last editing action", triggered=self.redo)
        
        #         self.menu_actions['fullscreen'] = QtGui.QAction(None,"&Full Screen",self,shortcut="Ctrl+f",
        #                                            statusTip="Run in full screen mode",triggered=self.full_screen)
        
        self.menu_actions['quit'] = QAction(QIcon(_resources_dir+'/images/redo_24.png'),
                                                   "&Quit", self, shortcut=QKeySequence.Quit,
                                                   statusTip="Quit the application", triggered=self.close)
        
        self.menu_actions['about'] = QAction(QIcon("hello_world"),"&About", self,
                                                    statusTip="Show the application's About box",
                                                    triggered=self.on_about)
        
        self.menu_actions['aboutQt'] = QAction(QIcon('hello_world'),"About &Qt", self,
                                                     statusTip="Show the Qt library's About box",
                                                     triggered=qApp.aboutQt)
    
    def _init_menus(self):
        get_items = lambda *keys: [self.menu_actions.get(k,None) for k in keys]
        
        # --------------------------------------------------------------------------- #
        self.file_menu = self.menuBar().addMenu("&File")
        items = get_items('quit')#,'save','about','save as')
        self.add_actions(self.file_menu, items)  

        # --------------------------------------------------------------------------- #
        #self.edit_menu = self.menuBar().addMenu("&Edit")
        #items = get_items('undo','redo')
        #self.add_actions(self.edit_menu, items)
        
        # --------------------------------------------------------------------------- #
        #self.view_menu = self.menuBar().addMenu("&View")
        #self.toolbar_menu = self.view_menu.addMenu('&Toolbars')
        #self.tabs_menu = self.view_menu.addMenu("&Tabs")
        
        # --------------------------------------------------------------------------- #
        #self.menuBar().addSeparator()
        
        # --------------------------------------------------------------------------- #
        self.help_menu = self.menuBar().addMenu("&Help")
        items = get_items('about','aboutQt','')
        self.add_actions(self.help_menu, items)
    
    def _init_status_bar(self):
        self.status_text = QLabel("startup")
        self.statusBar().addWidget(self.status_text, 1)
         
    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)    
    
    def on_about(self):
        msg = """
        Thimbles is a set of python modules for handling spectra.
        This program is a GUI built on top of the Thimbles libraries.
        
        developed in the Cosmic Origins group at the University of Utah
        """
        QMessageBox.about(self, "about Thimbles GUI", msg)
