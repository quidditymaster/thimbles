import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')

import scipy.optimize

try: 
    from PySide import QtCore,QtGui
    from PySide.QtCore import *
    from PySide.QtGui import *
    matplotlib.rcParams['backend.qt4'] = 'PySide'
except ImportError:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    matplotlib.rcParams['backend.qt4'] = 'PyQt4'

import matplotlib.pyplot as plt
from  models import *
from views import *
from widgets import *

import thimbles as tmb
_resources_dir = os.path.join(os.path.dirname(__file__),"resources")

# ########################################################################### #

class AppForm(QMainWindow):
    
    def __init__(self, options):
        super(AppForm, self).__init__()
        self.setWindowTitle("Thimbles")
        self.main_frame = QWidget()        
        self.options = options
        
        self.layout = QHBoxLayout()
        self.main_table_model = MainTableModel()
        
        self.rfunc = eval("tmb.io.%s" % options.read_func)
        
        for sfile_name in options.spectra_files:
            try:
                spec_list, spec_inf = self.rfunc(sfile_name)
                if options.norm == "auto":
                    for spec in spec_list:
                        spec.approx_norm()
                for spec in spec_list:
                    spec.set_rv(options.rv)
                base_name = os.path.basename(sfile_name)
                spec_row = SpectraRow(spec_list, base_name)
                self.main_table_model.addRow(spec_row)
            except Exception as e:
                print "there was an error reading file %s" % sfile_name
                print e
        
        if options.line_list != None:
            try:
                ldat = np.loadtxt(options.line_list ,skiprows=1, usecols=[0, 1, 2, 3])
                base_name = os.path.basename(options.line_list)
                ll_row = LineListRow(ldat, base_name)
            except Exception as e:
                print "there was an error reading file %s" % options.line_list
                print e
        
        #setup for the dual spectrum operations
        self.partial_result = None
        self.current_operation = None

        self.main_table_view = NameTypeTableView(self)
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
        self._init_actions()
        self._init_menus()
        
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
        self.fit_features_btn = QPushButton("fit features")
        #self.tell_btn = QPushButton("extract telluric")
        btn_grid.addWidget(self.load_btn, 0, 0, 1, 1)
        btn_grid.addWidget(self.norm_btn, 1, 0, 1, 1)
        btn_grid.addWidget(self.fit_features_btn, 2, 0, 1, 1)
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
        self.eq_btn = QPushButton("=")
        btn_grid.addWidget(self.add_btn, 0, 0, 1, 1)
        btn_grid.addWidget(self.sub_btn, 0, 1, 1, 1)
        btn_grid.addWidget(self.mul_btn, 1, 0, 1, 1)
        btn_grid.addWidget(self.div_btn, 1, 1, 1, 1)
        btn_grid.addWidget(self.eq_btn, 2, 0, 1, 2)
        op_box.setLayout(btn_grid)
        return op_box
    
    def _init_multi_operations(self):
        op_box = QGroupBox("multi spectrum operations")
        btn_grid = QGridLayout()
        self.coadd_btn = QPushButton("coadd")
        btn_grid.addWidget(self.coadd_btn, 0, 0, 1, 1)
        op_box.setLayout(btn_grid)
        return op_box
    
    def _connect(self):
        self.main_table_view.doubleClicked.connect(self.on_double_click)
        self.div_btn.clicked.connect(self.on_div)
        self.eq_btn.clicked.connect(self.on_eq)
        self.load_btn.clicked.connect(self.on_load_spectrum)
        self.fit_features_btn.clicked.connect(self.on_fit_features)
    
    def get_row(self, row):
        return self.main_table_model.rows[row]
    
    def on_double_click(self, index):
        row_index = index.row()
        row_object = self.get_row(row_index)
        row_object.on_double_click()
    
    def on_div(self):
        smod = self.main_table_view.selectionModel()
        selrows = smod.selectedRows()
        if len(selrows) != 1:
            #self.statusBar.setText("one at a time!")
            return
        row = selrows[0].row()
        if self.main_table_model.types[row] != "spectra":
            #self.status_bar.setText("spectra only!")
            return
        else:
            self.partial_result = self.main_table_model.internalData(row)
            self.current_operation = "/"
    
    def on_eq(self):
        smod = self.main_table_view.selectionModel()
        selrows = smod.selectedRows()
        if len(selrows) != 1:
            #self.statusBar.setText("one at a time!")
            return
        row = selrows[0].row()
        if self.main_table_model.types[row] != "spectra":
            #self.status_bar.setText("spectra only!")
            return
        else:
            if self.partial_result != None:
                second_operand = self.main_table_model.internalData(row)
                n2 = len(second_operand)
                n1 = len(self.partial_result)
                if n1 == 1 or n2 == 1:
                    match_type = "one to many"
                elif n1 == n2:
                    match_type = "ordering"
                else:
                    print "unable to match spectra"
                    return
                if match_type == "ordering":
                    for pair_idx in range(n1):
                        left_spec = self.partial_result[pair_idx]
                        right_spec = second_operand[pair_idx]
                    
    def on_fit_features(self):
        smod = self.main_table_view.selectionModel()
        selrows = smod.selectedRows()
        if len(selrows) != 2:
            #self.statusBar.setText("one at a time!")
            return
        row1, row2 = selrows[0].row(), selrows[1].row()
        spec = None
        ll = None
        for row_index in [row1, row2]:
            row = self.get_row(row_index)
            if row.type_id == "spectra":
                spec = row.data
                spec_name = row.name
            elif self.main_table_model.types[row] == "line list":
                ll = row.data
                ll_name = row.name
        if spec != None and ll != None:
            culled, feat_spec_idxs = self.cull_lines(spec, ll)
            if len(culled) == 0:
                print "no features survived the culling! check your wavelength solution"
                return
            fit_features = self.initial_feature_fit(spec, culled, feat_spec_idxs)
            frow = FeaturesRow(fit_features, "")
            features_name = "%s %s features" % (spec_name, ll_name)
            self.main_table_model.addRow((spec, fit_features, feat_spec_idxs, options.fwidth)) 
    
    def on_set_rv(self):
        pass
    
    def on_load_spectrum(self):
        fname, filters = QFileDialog.getOpenFileName(self, "load spectrum")
        lspec, inf = self.rfunc(fname)
        try:
            spec_list, spec_inf = self.rfunc(fname)
            if options.norm == "auto":
                for spec in spec_list:
                    spec.approx_norm()
            base_name = os.path.basename(fname)
            self.main_table_model.addItem(base_name, "spectra", lspec)
        except Exception as e:
            print "there was an error reading file %s" % fname
            print e 
    
    
    def cull_lines(self, spectra, ldat):
        new_ldat = []
        accepted_mask = np.zeros(len(ldat), dtype=bool)
        line_spec_idxs = np.zeros(len(ldat), dtype=int)
        for spec_idx in range(len(spectra)):
            spec = spectra[spec_idx]
            min_wv = spec.wv[0]
            max_wv = spec.wv[-1]
            for feat_idx in range(len(ldat)):
                cwv, cid, cep, cloggf = ldat[feat_idx]
                if (min_wv + 0.1) < cwv < (max_wv-0.1):
                    accepted_mask[feat_idx] = True
                    line_spec_idxs[feat_idx] = spec_idx
        for feat_idx in range(len(ldat)):
            if accepted_mask[feat_idx]:
                cwv, cid, cep, cloggf = ldat[feat_idx]
                new_ldat.append((cwv, cid, cep, cloggf))
        new_ldat = np.array(new_ldat)
        return new_ldat, line_spec_idxs
    
    def initial_feature_fit(self, spectra, ldat, feat_spec_idxs):
        features = []
        for feat_idx in range(len(ldat)):
            print "fitting feature", feat_idx + 1
            cwv, cid, cep, cloggf = ldat[feat_idx]
            spec = spectra[feat_spec_idxs[feat_idx]]
            bspec = spec.bounded_sample((cwv-0.25, cwv+0.25))
            if bspec == None:
                continue
            wvs = bspec.wv
            flux = bspec.flux
            norm = bspec.norm
            nflux = flux/norm
            
            tp = tmb.features.AtomicTransition(cwv, cid, cloggf, cep)
            start_p = np.array([0.0, 0.1, 0.0])
            lprof = tmb.line_profiles.Voigt(cwv, start_p)
            eq = 0.001
            nf = tmb.features.Feature(lprof, eq, 0.00, tp)
            
            wv_del = (wvs[-1]-wvs[0])/float(len(wvs))
            def resids(pvec):
                ew=pvec[0]
                pr = lprof.get_profile(wvs, pvec[1:])
                lsig = pvec[2]
                coff = pvec[1]
                sig_reg = 0
                if lsig < 0.5*wv_del:
                    sig_reg = 20.0*(lsig-0.5*wv_del)
                return np.hstack(((nflux - 1.0)+ew*pr, sig_reg))
            
            guessv = np.hstack((0.05, start_p))
            fit_res = fit_feature = scipy.optimize.leastsq(resids, guessv)
            fit = fit_res[0]
            fit[2:] = np.abs(fit[2:])
            lprof.set_parameters(fit[1:])
            nf.set_eq_width(fit[0]) 
            features.append(nf)
        return features
    
    def _init_fit_widget(self):
        self.fit_widget = FeatureFitWidget(self.spec, self.features, 0, self.options.fwidth, parent=self)
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 1)
    
    def save (self):
        QMessageBox.about(self, "Save MSG", "SAVE THE DATA\nTODO")

    def undo (self):
        QMessageBox.about(self, "Undo", "UNDO THE DATA\nTODO")
    
    def redo (self):
        QMessageBox.about(self, "Redo", "REDO THE DATA\nTODO")

    def _init_actions(self):

        self.menu_actions = {}
        
        self.menu_actions['save'] = QtGui.QAction(QtGui.QIcon(_resources_dir+'/images/save.png'),
                "&Save...", self, shortcut=QtGui.QKeySequence.Save,
                statusTip="Save the current data",
                triggered=self.save)
        
        self.menu_actions['save as'] = QtGui.QAction(QtGui.QIcon(_resources_dir+'/images/save_as.png'),
                "&Save As...", self, shortcut=QtGui.QKeySequence.SaveAs,
                statusTip="Save the current data as....",
                triggered=self.save)
        
        self.menu_actions['undo'] = QtGui.QAction(QtGui.QIcon(_resources_dir+'/images/undo_24.png'),
                "&Undo", self, shortcut=QtGui.QKeySequence.Undo,
                statusTip="Undo the last editing action", triggered=self.undo)
        
        self.menu_actions['redo'] =  QtGui.QAction(QtGui.QIcon(_resources_dir+'/images/redo_24.png'),
                                                   "&Redo", self, shortcut=QtGui.QKeySequence.Redo,
                                                   statusTip="Redo the last editing action", triggered=self.redo)
        
        #         self.menu_actions['fullscreen'] = QtGui.QAction(None,"&Full Screen",self,shortcut="Ctrl+f",
        #                                            statusTip="Run in full screen mode",triggered=self.full_screen)
        
        self.menu_actions['quit'] = QtGui.QAction(QtGui.QIcon(_resources_dir+'/images/redo_24.png'),
                                                   "&Quit", self, shortcut=QtGui.QKeySequence.Quit,
                                                   statusTip="Quit the application", triggered=self.close)
        
        self.menu_actions['about'] = QtGui.QAction(QtGui.QIcon("hello_world"),"&About", self,
                                                    statusTip="Show the application's About box",
                                                    triggered=self.on_about)
        
        self.menu_actions['aboutQt'] = QtGui.QAction(QtGui.QIcon('hello_world'),"About &Qt", self,
                                                     statusTip="Show the Qt library's About box",
                                                     triggered=QtGui.qApp.aboutQt)
    
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

class MainApplication (QApplication):
    """
    TODO: write doc string
    """
    
    def __init__ (self,options):
        super(MainApplication,self).__init__([])
        self.aboutToQuit.connect(self.on_quit)
        screen_rect = self.desktop().screenGeometry()
        size = screen_rect.width(), screen_rect.height()
        # TODO: use size to make main window the full screen size
        self.main_window = AppForm(options)
        self.main_window.show()
    
    def on_quit (self):
        pass

def main(options):
    try:
        app = MainApplication(options)
    except RuntimeError:
        app = MainApplication.instance()
    sys.exit(app.exec_())

if __name__ == "__main__":
    import argparse
    desc = "a spectrum processing and analysis GUI"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("spectra_files", nargs="*", help="paths to one or more spectrum data files")
    parser.add_argument("-line_list", "-ll", help="the path to a linelist file to load")
    parser.add_argument("-fwidth", "-fw",  type=float, default=3.0, 
                        help="the number of angstroms on either side of the current feature to display while fitting")
    parser.add_argument("-read_func", default="read_fits")
    parser.add_argument("-rv", type=float, default=0.0, help="optional radial velocity shift to apply")
    #parser.add_argument("-order", type=int, default=0, help="if there are multiple spectra specify which one to pull up")
    parser.add_argument("-norm", default="ones", help="how to normalize the spectra on readin options are ones and auto' ")
    options = parser.parse_args()
    
    main(options)
