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


# define default widget for a given key?
class TaskArgInput (QtGui.QWidget):
    
    def __init__(self,argname,default_argstring="",parent=None):
        QtGui.QWidget.__init__(self,parent)  
        self.argname = argname
        self.default_argstring = default_argstring
        self.initUI()
     
    def initUI (self):          
        # argname  [ argstring ]
        layout = QtGui.QHBoxLayout()        
        layout.addWidget(QtGui.QLabel(self.argname))
        self.argstring = QtGui.QLineEdit(self.default_argstring,parent=self)
        layout.addWidget(self.argstring)
        self.setLayout(layout)
                         
    def get_argstring (self):
        return self.argstring.text()

class TaskDialog (QtGui.QDialog):
    
    def __init__ (self,task):
        self.task = task    
        QtGui.QDialog.__init__(self)
        self.initUI()
        self.output_name = None
        self.output_values = None
        
        
        
    def initUI (self):        
        layout = QtGui.QVBoxLayout() 
        self.argstring_widgets = {}       
        for argname in self.task.argstrings:
            input = self.task.create_arg_widget(argname)                                
            self.argstring_widgets[argname] = input
            layout.addWidget(input)
        
        # TODO: make this it's own special text box
        self.output_name_dialog = TaskArgInput("Task Return Name","{}_result".format(self.task.task_name))
        layout.addWidget(self.output_name_dialog)
        
        self.run_btn = QtGui.QPushButton("Run")
        self.run_btn.clicked.connect(self.run)
        
        self.cancel_btn = QtGui.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel)
        
        self.help_btn = QtGui.QPushButton("Help")
        
        
        qgroupbox = QtGui.QGroupBox()
        hl = QtGui.QHBoxLayout()
        qgroupbox.setLayout(hl)
        hl.addWidget(self.run_btn)        
        hl.addWidget(self.cancel_btn)              
        hl.addWidget(self.help_btn)        
        layout.addWidget(qgroupbox)
                
        self.setLayout(layout)
        
    def run (self):    
        for argname in self.argstring_widgets:
            argstring = self.argstring_widgets[argname].get_argstring()
            self.task.argstrings[argname] = argstring
        
        try:
            self.output_values = self.task.eval()
        except EvalError as e:
            # TODO: handle this better
            e.argname
            raise e                
        self.output_name = self.output_name_dialog.get_argstring()        
        self.accept()
                
    def cancel (self):
        self.reject()
                
class PyQtTask (Task):


    def __init__ (self,target,argstrings=None,widgets=None,task_name=None):
        if isinstance(target,Task):
            Task.__init__(self,target.target,argstrings=target.argstrings)    
        else:
            Task.__init__(self,target,argstrings=argstrings)    
        # dictionary of widgets
        if widgets is None:
            widgets = {}
        self.widgets = widgets
        # set a specific task name
        if task_name is None:
            task_name = self.target.__name__
        self.task_name = task_name
      
    def create_arg_widget_class (self,key):
        """ """
        if key not in self.widgets:
            self.widgets[key] = TaskArgInput
        return self.widgets[key]        
    
    def create_arg_widget (self,key):
        WidgetClass = self.create_arg_widget_class(key)        
        return WidgetClass(key,self.argstrings.get(key,""))
     
    def create_task_dialog (self):
        return TaskDialog(self)
            
    def connect_argstrings (self,widget):
        """ connect all the container changes to change self.input """
        pass

    def help_widget (self):
        help_str = self.help()
        # create widget with this text
        widget = None
        return widget
    
    def on_eval (self):
        # ?? maybe ?? when event is evaluated
        pass


