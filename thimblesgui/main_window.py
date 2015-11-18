
# standard library
from itertools import cycle, product
import os
import sys
import time
import pickle

# 3rd party packages
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize

#internal imports
from thimblesgui import QtCore, QtGui, Qt
from thimbles.options import opts
from thimbles.tasks import task_registry
from thimbles import workingdataspace as wds

import thimblesgui as tmbg
from thimblesgui.views import ObjectTreeWidget
from thimblesgui.task_widgets import TaskLauncher 
import thimbles as tmb
from thimblesgui.active_collections import MappedListModel, ActiveCollection, ActiveCollectionView
from thimblesgui.object_creation_dialogs import NewStarDialog

from thimbles import ThimblesDB
gui_resource_dir = os.path.join(os.path.dirname(__file__),"resources")

# ########################################################################### #


class ThimblesMainWindow(QtGui.QMainWindow):
    
    def __init__(self, project_db_path):
        super(ThimblesMainWindow, self).__init__()
        self.setWindowTitle("Thimbles")
        self.db = ThimblesDB(project_db_path)
        self.selection = tmbg.selection.GlobalSelection(
            channels=[
                "source",
                "spectrum",
                "rv",
                "norm",
            ],
        )
        tmb.wds.gui_selection = self.selection
        
        self.make_actions()
        self.make_menus()
        self.make_tool_bar()
        self.make_status_bar()
        self.make_dock_widgets()
    
    def make_actions(self):
        #QtGui.QAction(QtGui.QIcon(":/images/new.png"), "&Attach Database", self)
        self.save_act = QtGui.QAction("&Save", self)
        self.save_act.setShortcut("Ctrl+s")
        self.save_act.setStatusTip("commit state to database")
        self.save_act.triggered.connect(self.on_save)
        
        self.quit_act = QtGui.QAction("&Quit", self)
        self.quit_act.setShortcut("Ctrl+Q")
        self.quit_act.setStatusTip("Quit Thimbles")
        self.quit_act.triggered.connect(self.close)
        
        self.new_star_act = QtGui.QAction("&New Star", self)
        self.new_star_act.setStatusTip("Create a New Star object")
        self.new_star_act.triggered.connect(self.on_new_star)
        
        self.about_act = QtGui.QAction("&About", self)
        self.about_act.setStatusTip("About Thimbles")
        self.about_act.triggered.connect(self.on_about)
    
    def make_menus(self):
        menu_bar = self.menuBar()
        
        self.file_menu = menu_bar.addMenu("&File")
        self.file_menu.addAction(self.quit_act)
        self.file_menu.addAction(self.save_act)
        
        self.context_menu = menu_bar.addMenu("&Context")
        source_menu = self.context_menu.addMenu("Sources")
        source_menu.addAction(self.new_star_act)
        
        self.help_menu = self.menuBar().addMenu("&Help")
        self.help_menu.addAction(self.about_act)
    
    def make_tool_bar(self):
        self.file_tool_bar = self.addToolBar("File")
        self.file_tool_bar.addAction(self.save_act)
    
    def make_status_bar(self):
        self.statusBar().showMessage("Ready")
    
    def make_dock_widgets(self):
        dock = QtGui.QDockWidget("sources", self)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        source_collection = ActiveCollection(self.db)
        self.source_widget = ActiveCollectionView(
            active_collection = source_collection,
            selection=self.selection,
            selection_channel="source",
        )
        dock.setWidget(self.source_widget)
        self.setCentralWidget(dock)

        dock = QtGui.QDockWidget("sources2", self)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        also_source_collection = ActiveCollection(self.db)
        self.source_widget = ActiveCollectionView(
            active_collection = also_source_collection,
            selection=self.selection,
            selection_channel="source",
        )
        dock.setWidget(self.source_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
        #dock = QtGui.QDockWidget("working data space", self)
        #dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        #self.wds_widget = ObjectTreeWidget(wds, parent=dock)
        #dock.setWidget(self.wds_widget)
        #self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        #self.setCentralWidget(dock)
        
        #dock = QtGui.QDockWidget("tasks", self)
        #self.task_launcher = TaskLauncher(parent=dock)
        #dock.setWidget(self.task_launcher)
        #self.addDockWidget(Qt.RightDockWidgetArea, dock)
    
    def bad_selection(self, msg=None):
        """indicate when operations cannot be performed because of bad user selections
        """
        if msg == None:
            msg = "invalid selection\n"
        else:
            msg = "invalid selection\n" + msg
        self.wd = tmbg.dialogs.WarningDialog(msg)
        self.wd.warn()
    
    def on_save(self):
        self.db.commit()
        save_msg = "committed to {}".format(self.db.path)
        self.statusBar().showMessage(save_msg)
    
    def on_new_star(self):
        new_star = NewStarDialog.get_new(parent=self)
        if not new_star is None:
            self.db.add(new_star)
    
    def on_about(self):
        about_msg =\
"""
THIMBLES:
  Tools for
  Handling 
  Integrated
  Modeling of
  Breathtakingly
  Large
  Ensembles of
  Spectra

Thimbles was developed in the Cosmic Origins group at the University of Utah.
"""
        QtGui.QMessageBox.about(self, "about Thimbles GUI", about_msg)
    
    def print_args(self, *args, **kwargs):
        """prints the arguments passed in
        useful for connecting to events during gui debugging"""
        print("in print_args")
        print(args, kwargs)

