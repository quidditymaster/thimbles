#this is the place to put dialogs for selecting

from PySide.QtGui import *
from PySide.QtCore import *
from models import *

import thimbles as tmb

class LoadDialog(QDialog):

    def __init__(self, parent=None):
        super(LoadDialog, self).__init__(parent)
        self.initUI()
    
    def initUI(self):
        lay = QGridLayout()
        self.fname_label = QLabel("file path")
        self.file_le = QLineEdit("", self)
        lay.addWidget(self.fname_label, 0, 0, 1, 1)
        lay.addWidget(self.file_le, 0, 1, 1, 1)
        self.browse_btn = QPushButton("Browse", self)
        lay.addWidget(self.browse_btn, 0, 2, 1, 1)
        self.name_label = QLabel("name")
        self.name_edited = False
        self.name_le = QLineEdit("", self)
        lay.addWidget(self.name_label, 1, 0, 1, 1)
        lay.addWidget(self.name_le, 1, 1, 1, 1)
        self.type_label = QLabel("object type")
        lay.addWidget(self.type_label, 2, 0, 1, 1)
        self.type_dd = QComboBox()
        self.type_ids = ["spectra", "line list"]
        self.type_dd.addItems(self.type_ids)
        lay.addWidget(self.type_dd, 2, 1, 1, 1)
        
        self.function_dd = QComboBox()
        spec_io_names = [f for f in dir(tmb.io) if "read_" in f]
        spec_io_funcs = map(lambda x: eval("tmb.io." + x), spec_io_names)
        
        ll_io_names = ["loadtxt"]
        ll_io_funcs = [lambda x: np.loadtxt(x, usecols=[0, 1, 2, 3])]
        self.function_dd.addItems(spec_io_names)
        
        self.load_btn = QPushButton("Load")
        self.cancel_btn = QPushButton("Cancel")
        lay.addWidget(self.load_btn, 4, 2, 1, 1)
        lay.addWidget(self.cancel_btn, 4, 1, 1, 1)
        
        self.load_func = tmb.io.read_fits
        
        tt = [("spectra", spec_io_funcs),
              ("line list", ll_io_funcs)
            ] 
        
        #do the event connections
        self.browse_btn.clicked.connect(self.on_browse)
        self.load_btn.clicked.connect(self.on_load)
        self.cancel_btn.clicked.connect(self.on_cancel)

        #set the layout
        self.setLayout(lay)

        self.new_row = None
    
    def on_browse(self):
        fname, filters = QFileDialog.getOpenFileName(self, "select file path")
        if fname:
            self.file_le.setText(fname)
    
    def on_set_type(self):
        pass
    
    def on_cancel(self):
        print "self.reject", self.reject
        self.reject()
    
    def on_load(self):
        try:
            fname = self.file_le.text()
            loaded_obj = self.load_func(fname)
            row_name = self.name_le.text()
            self.new_row = SpectraRow(loaded_obj, row_name)
            self.accept()
        except:
            qmb = QMessageBox()
            qmb.setText("There was a problem reading the file")
            qmb.exec_()
    
    def get_row(self):
        self.exec_()
        return self.new_row

if __name__ == "__main__":
    qap = QApplication([])
    
    ld = LoadDialog()
    res = ld.get_row()
    print "res", res
    
    #qap.exec_()
