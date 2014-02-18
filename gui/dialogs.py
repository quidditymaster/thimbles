import os

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
        self.function_label = QLabel("readin function")
        lay.addWidget(self.function_label, 3, 0, 1, 1)
        lay.addWidget(self.function_dd, 3, 1, 1, 1)
        spec_io_names = [f for f in dir(tmb.io) if "read" in f]
        spec_io_funcs = map(lambda x: eval("tmb.io." + x), spec_io_names)
        
        ll_io_names = ["loadtxt"]
        ll_io_funcs = [lambda x: np.loadtxt(x, usecols=[0, 1, 2, 3])]
        
        self.loading_functions = {}
        self.loading_function_names = {}
        self.loading_functions["spectra"] = spec_io_funcs
        self.loading_functions["line list"] = ll_io_funcs
        self.loading_function_names["spectra"] = spec_io_names
        self.loading_function_names["line list"] = ll_io_names
        
        self.on_type_changed()
        
        self.load_btn = QPushButton("Load")
        self.cancel_btn = QPushButton("Cancel")
        lay.addWidget(self.load_btn, 4, 2, 1, 1)
        lay.addWidget(self.cancel_btn, 4, 1, 1, 1)
        
        #do the event connections
        self.function_dd.currentIndexChanged.connect(self.on_function_changed)
        self.type_dd.currentIndexChanged.connect(self.on_type_changed)
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
        if not self.name_le.text():
            base_name = os.path.basename(fname)
            self.name_le.setText(base_name)
    
    def on_type_changed(self):
        self.cur_type_id = self.type_dd.currentText()
        self.function_dd.clear()
        func_names = self.loading_function_names[self.cur_type_id]
        self.function_dd.addItems(func_names)
        self.on_function_changed()
    
    def on_function_changed(self):
        cur_type = self.cur_type_id
        function_index = self.function_dd.currentIndex()
        self.load_func = self.loading_functions[cur_type][function_index]
    
    def on_cancel(self):
        self.reject()
    
    def on_load(self):
        try:
            fname = self.file_le.text()
            loaded_obj = self.load_func(fname)
            row_name = self.name_le.text()
        except:
            qmb = QMessageBox()
            qmb.setText("There was a problem reading the file")
            qmb.exec_()
            return
        if self.type_dd.currentText() == "spectra":
            self.new_row = SpectraRow(loaded_obj, row_name)
        elif self.type_dd.currentText() == "line list":
            self.new_row = LineListRow(loaded_obj, row_name)
        self.accept()
    
    def get_row(self):
        self.exec_()
        return self.new_row



class RVSettingDialog(QDialog):

    def __init__(self, spectra, parent=None):
        super(RVSettingDialog, self).__init__(parent)
        self.spectra = spectra
        self.current_rv = spectra[0].get_rv()
        self.original_rv = self.current_rv
        self.initUI()
    
    def reset_rv_text(self):
        rvtext = "%5.3f" % self.current_rv
        self.rv_le.setText(rvtext)
    
    def initUI(self):
        self.rv_label = QLabel("radial velocity")
        self.units_label = QLabel("Km/S")
        self.rv_le = QLineEdit("", self)
        self.reset_rv_text()
        self.apply_btn = QPushButton("Apply")
        self.cancel_btn = QPushButton("Cancel")
        self.finish_btn = QPushButton("finish")
        
        #do the layout
        lay = QGridLayout()
        
        #row1
        lay.addWidget(self.rv_label, 0, 0, 1, 1)
        lay.addWidget(self.rv_le, 0, 1, 1, 1)
        lay.addWidget(self.units_label, 0, 2, 1, 1)
        
        #row2
        lay.addWidget(self.cancel_btn, 1, 0, 1, 1)
        lay.addWidget(self.apply_btn, 1, 1, 1, 1)
        lay.addWidget(self.finish_btn, 1, 2, 1, 1)
        
        #connect
        self.rv_le.editingFinished.connect(self.on_rv_le_changed)
        self.apply_btn.clicked.connect(self.on_apply)
        self.cancel_btn.clicked.connect(self.on_cancel)
        self.finish_btn.clicked.connect(self.on_finish)
        
        self.setLayout(lay)
    
    def on_rv_le_changed(self):
        try:
            new_rv_val = float(self.rv_le.text())
            self.current_rv = new_rv_val
        except ValueError:
            pass
        self.reset_rv_text()
    
    def set_rv(self):
        self.show()
    
    def on_apply(self):
        for spectrum in self.spectra:
            spectrum.set_rv(self.current_rv)
    
    def on_cancel(self):
        for spectrum in self.spectra:
            spectrum.set_rv(self.original_rv)
        self.reject()
    
    def on_finish(self):
        self.on_apply()
        self.accept()

if __name__ == "__main__":
    qap = QApplication([])
    
    ld = LoadDialog()
    res = ld.get_row()
    print "res", res
    
    #qap.exec_()
