#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE: For the basic tests of Thimbles task widgets
AUTHOR: dylangregersen
DATE: Mon Aug 25 14:45:46 2014
"""
# ########################################################################### #

# import modules


import os
import sys
import re
import time
import numpy as np
from thimblesgui import QtCore,QtGui,Qt
try:
    import ipdb as pdb
except ImportError:
    import pdb
from thimbles.tasks import Task
from thimblesgui.task_widgets import *

# ########################################################################### #

class ListTasksWidget (QtGui.QWidget):
    """ display a list of tasks which can be clicked to create task widgets 
    
    Maybe also show the output of the task somehow
    
    """        
    def __init__(self,tasks):
        QtGui.QWidget.__init__(self)         
        self.tasks = {}
        for t in tasks:
            pt = PyQtTask(t,task_name=t.target.__name__.upper())
            self.tasks[pt.task_name] = pt
        self.initUI()
        self.junk = []
    
    def initUI(self):                               
        layout = QtGui.QVBoxLayout()
        # list of tasks
        for name in self.tasks:
            t = self.tasks[name]
            qbtn = QtGui.QPushButton(name, self)
            qbtn.clicked.connect(self.create_dialog(name))            
            layout.addWidget(qbtn)
            
            # TODO: create task widget when clicked
            #qbtn.clicked.connect(QtCore.QCoreApplication.instance().quit)
            # qbtn.resize(qbtn.sizeHint())
            # qbtn.move(50, 50)       
        
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Tasks')    
    
        self.setLayout(layout)
    
    def create_dialog (self,name):
        def junky ():
            def create ():
                dialog = self.tasks[name].create_task_dialog()    
                dialog.show()
                return dialog 
                
            dialog = create()
            dialog.exec_()           
            print(dialog.output_name)
            print(dialog.output_values)
            # TODO: Store these into appropriate namespace (USERNAMESPACE)
            self.junk.append(dialog)
            print(self.junk)
        return junky
            
class MainApplication (QtGui.QApplication):

    def __init__ (self):
        QtGui.QApplication.__init__(self,[])
               
def func1 (x,y,a=1.0,b=2.0):
    """ Evaluates a*x+b*y """
    return a*x+b*y   
    
def add_values (x,y):
    """ adds x+y """
    return x+y     
                
def main():
    tasks = []
    tasks.append(Task(func1)) 
    tasks.append(Task(add_values)) 
    # TODO: create some tasks more
       
    try:
        app = MainApplication()
    except RuntimeError:
        app = MainApplication.instance()
    widget = ListTasksWidget(tasks)
    widget.show()
    sys.exit(app.exec_())

pass
# ########################################################################### #
if __name__ == '__main__':
    main()


