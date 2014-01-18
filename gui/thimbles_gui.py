import sys

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
        
        self.loaded_spectra, info = tmb.io.read_fits(options.spectrum_file)
        self.ldat = np.loadtxt(self.options.line_list, usecols=[0, 1, 2, 3])
        self._init_features()
        self._init_feature_table()
        self._init_fit_widget()
        
        self.main_frame.setLayout(self.layout)
        
        self.setCentralWidget(self.main_frame)
        
        self.create_menu()
        #self.create_main_frame()
        self.create_status_bar()
    
    def _init_features(self):
        self.features = []
        for feat_idx in range(len(self.ldat)):
            cwv, cid, cep, cloggf = self.ldat[feat_idx]
            tp = tmb.features.AtomicTransition(cwv, cid, cloggf, cep)
            lprof = tmb.line_profiles.Voigt(cwv, np.array([0.0, 0.01, 0.0]))
            eq = 0.001
            nf = tmb.features.Feature(lprof, eq, 0.00, tp)
            self.features.append(nf)

    def _init_fit_widget(self):
        res_min_wv = self.loaded_spectra[0].wv[-1]
        for f_idx in range(len(self.features)):
            if self.features[f_idx].wv > res_min_wv:
                break
        self.fit_widget = FeatureFitWidget(self.loaded_spectra[0], self.features, f_idx+1, parent=self)
        self.layout.addWidget(self.fit_widget, 2, 0, 2, 5)

    def _init_feature_table(self):
        drole = Qt.DisplayRole
        crole = Qt.CheckStateRole
        wvcol = Column("wavelength", getter_dict = {drole: lambda x: "%10.3f" % x.wv})
        spcol = Column("species", getter_dict = {drole: lambda x: "%10.3f" % x.species})
        epcol = Column("excitation\npotential", {drole: lambda x:"%10.3f" % x.ep})
        loggfcol = Column("log(gf)", {drole: lambda x: "%10.3f" % x.loggf})        
        offsetcol = Column("offset", {drole: lambda x: "%10.3f" % x.get_offset()})
        depthcol = Column("depth", {drole: lambda x: "%10.3f" % x.depth})
        viewedcol = Column("viewed", {crole: lambda x: x.flags["viewed"]}, checkable=True)
        #ewcol = Column("depth"
        columns = [wvcol, spcol, epcol, loggfcol, offsetcol, depthcol,
               viewedcol]
        self.linelist_model = ConfigurableTableModel(self.features, columns)
        self.linelist_view = LineListView(parent=self)
        self.linelist_view.setModel(self.linelist_model)
        self.layout.addWidget(self.linelist_view, 0, 0, 2, 3)
    
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
    options = parser.parse_args()
    
    main(options)
