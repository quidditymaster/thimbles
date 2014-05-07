import os
import numpy as np
import matplotlib.pyplot as plt

from PySide.QtGui import *
from PySide.QtCore import *

from options import options
import thimblesgui as tmbg
from thimblesgui import user_namespace
import thimbles as tmb

class SaveDialog(QDialog):
    
    def __init__(self, data, parent=None):
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
        spec_io_names = [f for f in dir(tmb.io.spec_io) if "read" in f]
        spec_io_funcs = [user_namespace.eval_(x) for x in spec_io_names]
        
        ll_io_names = [f for f in dir(tmb.io.linelist_io) if "read" in f]
        ll_io_funcs = [user_namespace.eval_(x) for x in ll_io_names] 
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
        fname, filters = QFileDialog.getSaveFileName(self, "select file path")
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
    
    def on_save(self):
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
            self.new_row = tmbg.models.SpectraRow(loaded_obj, row_name)
        elif self.type_dd.currentText() == "line list":
            self.new_row = tmbg.models.LineListRow(loaded_obj, row_name)
        self.accept()
    
    def save_row(self):
        self.exec_()

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
        spec_io_names = [f for f in dir(user_namespace) if "read_" in f]
        spec_io_funcs = [user_namespace.eval_(x) for x in spec_io_names]
        
        ll_io_names = ["loadtxt"]
        
        def skipload(x):
            try:
                res = np.loadtxt(x, usecols=[0, 1, 2, 3])
            except:
                res = np.loadtxt(x, usecols=[0, 1, 2, 3], skiprows=1)
            return res
        
        ll_io_funcs = [skipload] 
        self.loading_functions = {}
        self.loading_function_names = {}
        self.defaults = {}
        self.loading_functions["spectra"] = spec_io_funcs
        self.loading_functions["line list"] = ll_io_funcs
        self.loading_function_names["spectra"] = spec_io_names
        self.loading_function_names["line list"] = ll_io_names
        self.defaults["spectra"] = "read_spec"
        self.defaults["line list"] = "loadtxt"
        
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
        default_func_name = self.defaults[self.cur_type_id]
        default_idx = func_names.index(default_func_name)
        self.function_dd.setCurrentIndex(default_idx)
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
        except Exception as e:
            self.wd=WarningDialog("error reading file %s" % fname, error_message=str(e), parent=None):
            self.wd.warn()
            return
        if self.type_dd.currentText() == "spectra":
            self.new_row = tmbg.models.SpectraRow(loaded_obj, row_name)
        elif self.type_dd.currentText() == "line list":
            self.new_row = tmbg.models.LineListRow(loaded_obj, row_name)
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
        self.ccor_btn = QPushButton("Cross Correlate")
        self.apply_btn = QPushButton("Apply")
        self.cancel_btn = QPushButton("Cancel")
        self.finish_btn = QPushButton("Finish")
        
        #do the layout
        lay = QGridLayout()
        
        #row1
        lay.addWidget(self.rv_label, 0, 0, 1, 1)
        lay.addWidget(self.rv_le, 0, 1, 1, 1)
        lay.addWidget(self.units_label, 0, 2, 1, 1)
        
        #row2
        lay.addWidget(self.ccor_btn, 1, 0, 1, 1)
        lay.addWidget(self.cancel_btn, 1, 1, 1, 1)
        lay.addWidget(self.apply_btn, 1, 2, 1, 1)
        lay.addWidget(self.finish_btn, 1, 3, 1, 1)
        
        #connect
        self.rv_le.editingFinished.connect(self.on_rv_le_changed)
        self.ccor_btn.clicked.connect(self.on_ccor)
        self.apply_btn.clicked.connect(self.on_apply)
        self.cancel_btn.clicked.connect(self.on_cancel)
        self.finish_btn.clicked.connect(self.on_finish)
        
        self.setLayout(lay)
    
    def on_ccor(self):
        ccor_rv = tmb.velocity.template_rv_estimate(self.spectra, delta_max=options.max_rv)
        self.current_rv = ccor_rv
        self.reset_rv_text()
    
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


class WarningDialog(QDialog):
    
    def __init__(self, message, error_message=None, parent=None):
        super(WarningDialog, self).__init__(parent)
        lay = QGridLayout()
        self.message_label = QLabel(message)
        lay.addWidget(self.message_label, 0, 0, 1, 3)
        if error_message != None:
            self.error_text_box = QPlainTextEdit()
            self.error_text_box.setPlainText(str(error_message))
            self.error_text_box.setReadOnly(True)
            lay.addWidget(self.error_text_box, 1, 0, 1, 3)
        self.ok_btn = QPushButton("acknowledged")
        lay.addWidget(self.ok_btn, 2, 1, 1, 1)
        self.setLayout(lay)
        self.ok_btn.clicked.connect(self.on_ok)
    
    def warn(self):
        self.exec_()
    
    def on_ok(self):
        self.accept()
        

class NormalizationDialog(QDialog):
    
    def __init__(self, spectra, parent=None):
        super(NormalizationDialog, self).__init__(parent=parent)
        self.spectra = spectra
        self.orig_norms = [spec.norm for spec in self.spectra]
        self.initUI()
    
    def initUI(self):
        #set up the widgets
        self.algorithm_label = QLabel("algorithm")
        self.algorithm_dd = QComboBox()
        algo1 = ["adaptive echelle", 
                 "partition_scale=300, poly_order=3, mask_kwargs={'n_layers':3, 'first_layer_width':101, 'last_layer_width':11, 'rejection_sigma':2.0}, smart_partition=False, alpha=4.0, beta=4.0, overwrite=True",
                 user_namespace.eval_("tmb.utils.misc.approximate_normalization")]
        self.algos = [algo1]
        algo_names = [al[0] for al in self.algos]
        self.algorithm_dd.addItems(algo_names)
        self.params_le = QLineEdit()
        self.params_le.setText(algo1[1])
        
        self.apply_btn = QPushButton("Apply")
        self.cancel_btn = QPushButton("Cancel")
        self.ok_btn = QPushButton("Ok")
        self.preview_btn = QPushButton("Preview")
        
        lay = QGridLayout()
        lay.addWidget(self.algorithm_label, 0, 0)
        lay.addWidget(self.algorithm_dd, 0, 1)
        lay.addWidget(self.params_le, 1, 0, 1, 4)
        lay.addWidget(self.apply_btn, 2, 2)
        lay.addWidget(self.preview_btn, 2, 1)
        lay.addWidget(self.cancel_btn, 2, 3)
        lay.addWidget(self.ok_btn, 2, 4)
        
        self.setLayout(lay)
        
        self.algorithm_dd.currentIndexChanged.connect(self.on_algorithm_change)
        self.apply_btn.clicked.connect(self.on_apply)
        self.ok_btn.clicked.connect(self.on_ok)
        self.cancel_btn.clicked.connect(self.on_cancel)
    
    def get_norm(self):
        self.exec_()
    
    def on_apply(self):
        self.apply_norm()
    
    def apply_norm(self):
        cind = self.algorithm_dd.currentIndex()
        for spec in self.spectra:
            try:
                par_dict = eval("dict(%s)" % self.params_le.text())
                self.algos[cind][2](spec, **par_dict)
            except Exception as e:
                print e
                print "oops, there was a problem carrying out the norm"
    
    def on_cancel(self):
        for spec_idx in range(len(self.spectra)):
            self.spectra[spec_idx].norm = self.orig_norms[spec_idx]
        self.reject()
    
    def on_ok(self):
        self.apply_norm()
        self.accept()
    
    def on_algorithm_change(self):
        cind = self.algorithm_dd.currentIndex()
        self.params_le.setText(self.algos[cind][1])

if __name__ == "__main__":
    qap = QApplication([])
    
    #ld = LoadDialog()
    #res = ld.get_row()
    #print "res", res
    
    #try:
    #    a, b = 0
    #except Exception as e:
    #    wd = WarningDialog("warning bad stuff happened!", e)
    #    wd.warn()
    #
    #wd = WarningDialog("bad selection")
    #wd.warn()
    
    #qap.exec_()
