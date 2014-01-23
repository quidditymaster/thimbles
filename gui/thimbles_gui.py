import sys

import numpy as np
import matplotlib
import scipy.optimize
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

from  models import *
from views import *
from widgets import *

import thimbles as tmb

class AppForm(QMainWindow):
    
    def __init__(self, options, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle("Thimbles")
        self.main_frame = QWidget()        
        self.options = options
        
        self.layout = QGridLayout()
        
        self.ldat = np.loadtxt(self.options.line_list, usecols=[0, 1, 2, 3])
        self.loaded_spectra, info = tmb.io.read_fits(options.spectrum_file)
        for spec in self.loaded_spectra:
            spec.set_rv(options.rv)
        self.cull_lines()
        self._init_features()
        self._init_fit_widget()
        
        self.main_frame.setLayout(self.layout)
        
        self.setCentralWidget(self.main_frame)
        
        self.create_menu()
        self.create_status_bar()
        #import pdb; pdb.set_trace()
        self._connect()
    
    def _connect(self):
        #connect all the events
        #self.linelist_view.doubleClicked.connect(self.fit_widget.set_feature)
        pass
    
    def cull_lines(self):
        min_wv = self.loaded_spectra[0].wv[0]
        max_wv = self.loaded_spectra[0].wv[-1]
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
            
            bspec = self.loaded_spectra[0].bounded_sample((cwv-0.25, cwv+0.25))
            wvs = bspec.wv
            flux = bspec.flux
            norm = bspec.norm
            nflux = flux/norm
            
            tp = tmb.features.AtomicTransition(cwv, cid, cloggf, cep)
            start_p = np.array([0.0, 0.1, 0.0])
            lprof = tmb.line_profiles.Voigt(cwv, start_p)
            eq = 0.001
            nf = tmb.features.Feature(lprof, eq, 0.00, tp)
            
            def resids(pvec):
                ew=pvec[0]
                pr = lprof.get_profile(wvs, pvec[1:])
                return (nflux - 1.0)+ew*pr
            
            guessv = np.hstack((0.05, start_p))
            fit_res = fit_feature = scipy.optimize.leastsq(resids, guessv)
            lprof.set_parameters(fit_res[0][1:])
            nf.set_eq_width(fit_res[0][0]) 
            self.features.append(nf)
    
    def _init_fit_widget(self):
        self.fit_widget = FeatureFitWidget(self.loaded_spectra[0], self.features, 0, self.options.fwidth, parent=self)
        self.layout.addWidget(self.fit_widget, 0, 0, 1, 1)
    
    def create_menu(self):
        self.file_menu = self.menuBar().addMenu("&File")
        quit_action = self.create_action("&Quit", slot=self.close, 
                        shortcut="Ctrl+Q", tip="exit the application")
        self.add_actions(self.file_menu, (None, None, quit_action))
        
        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.create_action("&About", slot=self.on_about)
        self.add_actions(self.help_menu, (about_action,))
    
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
    
    def create_status_bar(self):
        self.status_text = QLabel("startup")
        self.statusBar().addWidget(self.status_text, 1)
    
    def on_about(self):
        msg = """
        Thimbles is a set of python modules for handling spectra.
        This program is a GUI built on top of the Thimbles libraries.
        
        developed in the Cosmic Origins group at the University of Utah
        """
        QMessageBox.about(self, "about Thimbles GUI", msg)


def main(options):
    app = QApplication([])
    form = AppForm(options)
    form.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    import argparse
    desc = "a spectrum processing and analysis GUI"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("spectrum_file", help="the path to a spectrum file")
    parser.add_argument("line_list", help="the path to a linelist file")
    parser.add_argument("-fwidth", "-fw",  type=float, default=6.0, 
                        help="the number of angstroms on either side of the current feature to display while fitting")
    parser.add_argument("-rv", type=float, default=0.0, help="star radial velocity shift")
    options = parser.parse_args()
    
    main(options)
