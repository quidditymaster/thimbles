import sys

import os
import numpy as np
import matplotlib
import scipy.optimize
matplotlib.use('Qt4Agg')
try: 
    from PySide import QtCore,QtGui
    from PySide.QtCore import *
    from PySide.QtGui import *
    matplotlib.rcParams['backend.qt4'] = 'PySide'
except ImportError:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    matplotlib.rcParams['backend.qt4'] = 'PyQt4'

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
        
        self.layout = QGridLayout()
        
        self.ldat = np.loadtxt(self.options.line_list, usecols=[0, 1, 2, 3])
        rfunc = eval("tmb.io.%s" % options.read_func)
        #self.loaded_spectra, info = rfunc(options.spectrum_file)
        loaded_spectra, info = rfunc(options.spectrum_file)
        self.spec = loaded_spectra[options.order]
        self.order_num = options.order
        #for spec in self.loaded_spectra:
        #    spec.set_rv(options.rv)
        self.spec.set_rv(options.rv)
        self.cull_lines()
        self._init_features()
        self._init_fit_widget()
        
        self.main_frame.setLayout(self.layout)
        self.setCentralWidget(self.main_frame)
        
        #self.create_menu()
        self._init_actions()
        self._init_menus()
        
        self._init_status_bar()
        #import pdb; pdb.set_trace()
        self._connect()
    
    def _connect(self):
        #connect all the events
        #self.linelist_view.doubleClicked.connect(self.fit_widget.set_feature)
        pass
    
    def cull_lines(self):
        min_wv = self.spec.wv[0]
        max_wv = self.spec.wv[-1]
        new_ldat = []
        for feat_idx in range(len(self.ldat)):
            cwv, cid, cep, cloggf = self.ldat[feat_idx]
            if (min_wv + 0.1) < cwv < (max_wv-0.1):
                new_ldat.append((cwv, cid, cep, cloggf))
        self.ldat = np.array(new_ldat)
    
    def _init_features(self):
        self.features = []
        for feat_idx in range(len(self.ldat)):
            cwv, cid, cep, cloggf = self.ldat[feat_idx]
            
            bspec = self.spec.bounded_sample((cwv-0.25, cwv+0.25))
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
                sig_reg = 0
                if lsig < 0.5*wv_del:
                    sig_reg = 20.0*(lsig-0.5*wv_del)
                return np.hstack(((nflux - 1.0)+ew*pr, sig_reg))
            
            guessv = np.hstack((0.05, start_p))
            fit_res = fit_feature = scipy.optimize.leastsq(resids, guessv)
            lprof.set_parameters(fit_res[0][1:])
            nf.set_eq_width(fit_res[0][0]) 
            self.features.append(nf)
    
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
        items = get_items('quit','save','about','save as')
        self.add_actions(self.file_menu, items)  

        # --------------------------------------------------------------------------- #
        self.edit_menu = self.menuBar().addMenu("&Edit")
        items = get_items('undo','redo')
        self.add_actions(self.edit_menu, items)
        
        # --------------------------------------------------------------------------- #
        self.view_menu = self.menuBar().addMenu("&View")
        self.toolbar_menu = self.view_menu.addMenu('&Toolbars')
        self.tabs_menu = self.view_menu.addMenu("&Tabs")
        
        # --------------------------------------------------------------------------- #
        self.menuBar().addSeparator()
        
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
    
    def create_action(self, text, slot=None, shortcut = None,
                      icon = None, tip = None, checkable = False,
                      signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action
    
    
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
    parser.add_argument("spectrum_file", help="the path to a spectrum file")
    parser.add_argument("line_list", help="the path to a linelist file")
    parser.add_argument("-fwidth", "-fw",  type=float, default=3.0, 
                        help="the number of angstroms on either side of the current feature to display while fitting")
    parser.add_argument("-read_func", default="read_fits")
    parser.add_argument("-rv", type=float, default=0.0, help="star radial velocity shift")
    parser.add_argument("-order", type=int, default=0, help="if there are multiple spectra specify which one to pull up")
    options = parser.parse_args()
    
    main(options)
