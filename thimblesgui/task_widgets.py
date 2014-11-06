#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE: For Task dialogs and interaction
AUTHOR: dylangregersen
DATE: Mon Aug 25 14:44:30 2014
"""
# ########################################################################### #

# import modules 

from __future__ import print_function, division
import os 
import sys 
import re 
import time
import numpy as np  
from PySide import QtCore,QtGui

# ########################################################################### #
from thimbles.options import OptionSpecificationError
import thimbles.workingdataspace as wds

# define default widget for a given key?
class OptionValueSpecifierWidget(QtGui.QWidget):
    
    def __init__(self,option,parent=None):
        QtGui.QWidget.__init__(self,parent)
        self.option = option  
        self.initUI()
    
    def initUI (self):          
        # argname  [ argstring ]
        label_row = 0
        rstr_row = 1
        val_row = 2
        label_col = 0
        le_col = 0
        btn_col = 1
        
        layout = QtGui.QGridLayout()
        layout.addWidget(QtGui.QLabel(self.option.name), label_row, label_col, 1, 1)
        
        #runtime string widget line
        runtime_str = self.option.runtime_str
        if runtime_str is None:
            runtime_str = ""
        self.runtime_le = QtGui.QLineEdit(runtime_str, parent=self)
        self.runtime_le.editingFinished.connect(self.on_eval)
        layout.addWidget(self.runtime_le, rstr_row, le_col, 1, 1)
        self.eval_btn = QtGui.QPushButton("eval")
        self.eval_btn.clicked.connect(self.on_eval)
        layout.addWidget(self.eval_btn, rstr_row, btn_col, 1, 1)
        
        #value repr widget line
        value_repr = self.get_value_repr()
        self.value_le = QtGui.QLabel(value_repr, parent=self)
        layout.addWidget(self.value_le, val_row, le_col, 1, 1)
        
        self.setLayout(layout)
    
    def get_value_repr(self):
        try:
            val = self.option.value
        except OptionSpecificationError as e:
            val = e
        value_repr = repr(val)
        return value_repr
    
    def on_eval(self):
        print("running on_eval")
        try:
            runtime_str = self.runtime_le.text()
            self.option.set_runtime_str(runtime_str)
            value_repr = self.get_value_repr()
            self.value_le.setText(value_repr)
        except Exception as e:
            print(e)
            pass #TODO: give feedback to the user somehow

def task_runner_factory(task, parent=None):
    def setup_and_run_task():
        rtd = RunTaskDialog(task, parent=parent)
        rtd.exec_()
        time.sleep(0.01)
    return setup_and_run_task

class RunTaskDialog(QtGui.QDialog):
    
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
        opt_list = child_options.values()
        for opt_idx, option in enumerate(opt_list):
            print("in option iter", option)
            opt_wid = OptionValueSpecifierWidget(option, parent=self)
            self.scroll_layout.addWidget(opt_wid, opt_idx, 0)
        
        self.scroll.setWidget(self.scroll_box)
        #TODO: figure out what the hell is the problem with adding the scroll widget
        layout.addWidget(self.scroll)
        
        # TODO: make this it's own special text box
        self.result_name_le = QtGui.QLineEdit(self.task.result_name)
        layout.addWidget(self.result_name_le)
        
        btn_group = QtGui.QWidget(parent=self)
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

