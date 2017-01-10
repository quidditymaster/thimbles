#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE: For Task dialogs and interaction
AUTHOR: dylangregersen
DATE: Mon Aug 25 14:44:30 2014
"""
# ########################################################################### #


import os 
import sys 
import re 
import time
import numpy as np  
from thimblesgui import QtCore, QtWidgets, QtGui, Qt

# ########################################################################### #
from thimbles.options import OptionSpecificationError
from thimbles.tasks import task_registry
import thimbles.workingdataspace as wds

class FilePathOptionEditor(QtWidgets.QWidget):
    
    def __init__(self, option, parent=None):
        QtWidgets.QWidget.__init__(self, parent=parent)
        self.option = option
        
        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)
        try:
            start_fname = self.option.value
        except:
            start_fname = ""
        self.file_line_edit = QtGui.QLineEdit(start_fname, parent=self)
        layout.addWidget(self.file_line_edit)
        
        self.browse_btn = QtGui.QPushButton("browse", parent=self)
        layout.addWidget(self.browse_btn)
        
        self.browse_btn.clicked.connect(self.on_browse)
    
    def on_browse(self):
        fname, filters = QtGui.QFileDialog.getOpenFileName(self, "select file")
        if fname:
            self.file_line_edit.setText(fname)
        self.set_option_value()
    
    def set_option_value(self):
        self.option.set_runtime_str(self.file_line_edit.text())

class LineOptionEditor(QtWidgets.QWidget):
    
    def __init__(self, option, parent=None):
        QtWidgets.QWidget.__init__(self, parent=parent)
        self.option = option
        runtime_str = self.option.runtime_str
        if runtime_str is None:
            runtime_str = ""
        self.le = QtGui.QLineEdit(runtime_str)
        self.le.editingFinished.connect(self.set_option_value)
        
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.le)
        self.setLayout(layout)
    
    def set_option_value(self):
        self.option.set_runtime_str(self.le.text())


class StringRepresentation(QtWidgets.QWidget):
    
    def __init__(self, option, parent=None):
        QtWidgets.QWidget.__init__(self, parent=parent)
        self.option = option
        self.label = QtGui.QLabel("", parent=self)
        
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.update_representation()
    
    def update_representation(self):
        try:
            val = self.option.value
            repr_str = repr(val)
        except OptionSpecificationError as ose:
            repr_str = "Specification Invalid"
        self.label.setText(repr_str)


# define default widget for a given key?
class OptionValueSpecifierWidget(QtWidgets.QWidget):
    _label_row = 0
    _label_col = 0
    _editor_row = _label_row + 1
    _editor_col = 0
    _editor_row_span = 2
    _editor_col_span = 2
    _repr_row = _editor_row + _editor_row_span
    _repr_col = 0
    _repr_row_span = 2
    _repr_col_span = 2
    _set_btn_row = _editor_row
    _set_btn_col = _editor_col + _editor_col_span
    _set_btn_row_span = 1
    _set_btn_col_span = 1
    _fetch_btns_row = _repr_row
    _fetch_btns_col = _repr_col + _repr_col_span
    _fetch_btns_row_span = 1
    _fetch_btns_col_span = 1
    
    def __init__(self,option,parent=None):
        QtWidgets.QWidget.__init__(self,parent=parent)
        self.option = option  
        self.initUI()
    
    def _make_label(self):
        return QtGui.QLabel(self.option.name, parent=self)
    
    def _make_editor(self):
        if self.option.editor_style == "file":
            return FilePathOptionEditor(self.option, parent=self)
        else:
            return LineOptionEditor(self.option, parent=self)
    
    def _make_representer(self):
        return StringRepresentation(self.option, parent=self)
    
    def _make_set_btn(self):
        return QtGui.QPushButton("Set")
    
    def _make_fetch_btns(self):
        return QtGui.QPushButton("WDS")
    
    def label_rect(self):
        return (self._label_row, self._label_col, 1, 1)
    
    def editor_rect(self):
        return (self._editor_row, self._editor_col, self._editor_row_span, self._editor_col_span)
    
    def representer_rect(self):
        return (self._repr_row, self._repr_col, self._repr_row_span, self._repr_col_span)
    
    def set_btn_rect(self):
        return (self._set_btn_row, self._set_btn_col, self._set_btn_row_span, self._set_btn_col_span)
    
    def fetch_btns_rect(self):
        return (self._fetch_btns_row, self._fetch_btns_col, self._fetch_btns_row_span, self._fetch_btns_col_span)
    
    def initUI (self):
        layout = QtGui.QGridLayout()
        
        label = self._make_label()
        layout.addWidget(label, *self.label_rect())
        
        self.op_editor = self._make_editor()
        layout.addWidget(self.op_editor, *self.editor_rect())
        
        self.op_representer = self._make_representer()
        layout.addWidget(self.op_representer, *self.representer_rect())
        
        self.fetch_btns = self._make_fetch_btns()
        layout.addWidget(self.fetch_btns, *self.fetch_btns_rect())
        
        self.set_btn = self._make_set_btn()
        layout.addWidget(self.set_btn, *self.set_btn_rect())
        self.set_btn.clicked.connect(self.on_set)
        
        self.setLayout(layout)
    
    def keyPressEvent(self, event):
        ekey = event.key()
        if (ekey == Qt.Key_Enter) or (ekey == Qt.Key_Return):
            self.on_set()
            return
        super(OptionValueSpecifierWidget, self).keyPressEvent(event)
    
    def get_value_repr(self):
        try:
            val = self.option.value
        except OptionSpecificationError as e:
            val = e
        value_repr = repr(val)
        return value_repr
    
    def on_set(self):
        print("running on_set")
        try:
            self.op_editor.set_option_value()
            self.op_representer.update_representation()
        except Exception as e:
            print(e)
            pass #TODO: give feedback to the user somehow


def task_runner_factory(task, parent=None):
    def setup_and_run_task():
        rtd = RunTaskDialog(task, parent=parent)
        rtd.exec_()
        if parent is None:
            time.sleep(0.01)
    return setup_and_run_task

class RunTaskDialog(QtWidgets.QDialog):
    
    def __init__ (self, task, parent=None):
        self.task = task
        QtGui.QDialog.__init__(self, parent=parent)
        self.initUI()
        #self.output_name = None
        #self.output_values = None
    
    def initUI (self):        
        layout = QtGui.QVBoxLayout()
        #task name label
        layout.addWidget(QtGui.QLabel("Task: {}".format(self.task.name)))
        
        self.scroll_box = QtGui.QGroupBox("Task Options")
        self.scroll_layout = QtGui.QGridLayout()
        self.scroll_box.setLayout(self.scroll_layout)
        self.scroll = QtGui.QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        
        child_options = self.task.children
        opt_list = list(child_options.values())
        n_opt = len(child_options)
        for opt_idx, option in enumerate(opt_list):
            #print("in option iter", option)
            opt_wid = OptionValueSpecifierWidget(option, parent=self)
            self.scroll_layout.addWidget(opt_wid, opt_idx, 0)
        
        self.scroll.setWidget(self.scroll_box)
        #TODO: figure out what the hell is the problem with adding the scroll widget
        layout.addWidget(self.scroll)
        
        # TODO: make this it's own special text box
        self.result_name_le = QtGui.QLineEdit(self.task.result_name)
        layout.addWidget(self.result_name_le)
        
        btn_group = QtWidgets.QWidget(parent=self)
        hl = QtGui.QHBoxLayout()
        btn_group.setLayout(hl)
        
        self.run_btn = QtGui.QPushButton("Run")
        self.run_btn.clicked.connect(self.run)
        
        self.cancel_btn = QtGui.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel)
        
        #self.help_btn = QtGui.QPushButton("Help")
        
        hl.addWidget(self.cancel_btn)                      
        hl.addWidget(self.run_btn)        
        #hl.addWidget(self.help_btn)        
        layout.addWidget(btn_group)
        
        self.setLayout(layout)
    
    def run (self):
        print("inside run()")
        try:
            self.task.run()
            self.accept()
        except Exception as e:
            print(e)
            pass
    
    def cancel (self):
        self.reject()


class TaskLauncher(QtWidgets.QWidget):
    
    def __init__(self, parent):
        super(TaskLauncher, self).__init__(parent)
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        self.task_buttons = []
        self.dialog_functions = []
        for cur_task in list(task_registry.values()):
            cbutton = QtGui.QPushButton(cur_task.name)
            layout.addWidget(cbutton)
            task_runner_func = task_runner_factory(cur_task, parent=self)
            cbutton.clicked.connect(task_runner_func)
            self.dialog_functions.append(task_runner_func)
            self.task_buttons.append(cbutton)
