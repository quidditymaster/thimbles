
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
from thimblesgui.loading_dialog import LoadDialog
from thimblesgui.column_sets import star_columns, full_transition_columns, base_transition_columns, spectrum_columns
from thimblesgui.mct_dialog import ModelComponentTemplateApplicationDialog as MCTApplicationDialog
from thimbles.contexts import model_spines as contextualizers

from thimbles import ThimblesDB
gui_resource_dir = os.path.join(os.path.dirname(__file__),"resources")

# ########################################################################### #

class TargetCollectionDialog(QtGui.QDialog):
    collection = None
    behavior = None
    
    def __init__(
            self,
            collections,
            parent
    ):
        super(TargetCollectionDialog, self).__init__(parent=parent)
        self.collections = collections
        layout = QtGui.QGridLayout()
        names_label = QtGui.QLabel("collection")
        layout.addWidget(names_label, 0, 0, 1, 1)
        
        names_cbox = QtGui.QComboBox()
        layout.addWidget(names_cbox)
        for collection_name in collections:
            names_cbox.addItem(collection_name)
        
        self.names_cbox = names_cbox
        layout.addWidget(names_cbox, 0, 1, 1, 1)
        
        behavior_cbox = QtGui.QComboBox()
        behavior_cbox.addItem("replace")
        behavior_cbox.addItem("extend")
        self.behavior_cbox = behavior_cbox
        layout.addWidget(behavior_cbox, 1, 1, 1, 1)
        
        ok_btn = QtGui.QPushButton("OK")
        ok_btn.clicked.connect(self.on_ok)
        layout.addWidget(ok_btn, 2, 1, 1, 1)
        
        self.setLayout(layout)
    
    def on_ok(self):
        coll_name = self.names_cbox.currentText()
        self.collection = self.collections[coll_name]
        self.behavior=self.behavior_cbox.currentText()
        self.accept()

            
class ThimblesMainWindow(QtGui.QMainWindow):
    
    def __init__(self, app):
        super(ThimblesMainWindow, self).__init__()
        self.db = app.project_db#ThimblesDB(project_db_path)
        app.finish_splash(self)
        
        self.setWindowTitle("Thimbles")
        self.make_collections()
        self.selection = tmbg.selection.GlobalSelection(
            channels=[
                "star",
                "spectrum",
                "transition",
                "parameter",
            ],
        )
        tmb.wds.gui = self
        tmb.wds.gui_selection = self.selection
        tmb.wds.db = self.db
        
        SharedParameterSpace =tmb.analysis.SharedParameterSpace
        gp_result = self.db.query(SharedParameterSpace).filter(SharedParameterSpace.name == "global").all()
        if len(gp_result) > 1:
            raise Exception("multiple global parameter spaces found")
        elif len(gp_result) == 0:
            gp = tmb.analysis.SharedParameterSpace("global")
            gp_result = [gp]
            self.db.add(gp)
        global_params = gp_result[0]
        contextualizers.add_global("global", global_params)
        
        self.make_actions()
        self.make_menus()
        self.make_tool_bar()
        self.make_status_bar()
        self.make_dock_widgets()
    
    def make_collections(self):
        self.collections = {}
        
        spectra_collection = ActiveCollection(name="spectra")
        self.collections["spectra"] = spectra_collection
        
        star_collection = ActiveCollection(name="stars",)
        self.collections["stars"] = star_collection
        
        default_tq = "db.query(Transition).join(Ion).filter(Transition.wv > 2000).filter(Transition.wv < 20000).offset(0).limit(1000)"
        transition_collection = ActiveCollection(name="transitions",)
        self.collections["transitions"] = transition_collection
    
    def make_actions(self):
        self.load_act = QtGui.QAction("&Load", self)
        self.load_act.setStatusTip("read data from file")
        self.load_act.triggered.connect(self.on_load)
        
        #QtGui.QAction(QtGui.QIcon(":/images/new.png"), "&Attach Database", self)
        
        #DB actions
        self.query_db_act = QtGui.QAction("&Query", self)
        self.query_db_act.setToolTip("Load Results of a Database Query into a named Collection")
        self.query_db_act.triggered.connect(self.on_query_db)
        
        self.commit_act = QtGui.QAction("&Commit", self)
        #self.save_act.setShortcut("Ctrl+s")
        self.commit_act.setStatusTip("commit state to database")
        self.commit_act.triggered.connect(self.on_commit)
        
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
        #self.load_menu = self.file_menu.addMenu("&Load")
        self.db_menu = menu_bar.addMenu("&DB")
        self.context_menu = menu_bar.addMenu("&Context")
        
        self.file_menu.addAction(self.load_act)
        self.file_menu.addAction(self.quit_act)
        
        self.db_menu.addAction(self.query_db_act)
        self.db_menu.addAction(self.commit_act)
        
        source_menu = self.context_menu.addMenu("Sources")
        source_menu.addAction(self.new_star_act)
        
        #self.modeling_menu = menu_bar.addMenu("&Modeling")
        #self.modeling_menu.addAction(self.apply_mct_act)
        
        self.help_menu = self.menuBar().addMenu("&Help")
        self.help_menu.addAction(self.about_act)
    
    def make_tool_bar(self):
        pass
        #self.file_tool_bar = self.addToolBar("File")
        #self.file_tool_bar.addAction(self.commit_act)
    
    def make_status_bar(self):
        self.statusBar().showMessage("Ready")
    
    def attach_as_dock(self, dock_name, widget, dock_area):
        dock = QtGui.QDockWidget(dock_name, self)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        dock.setWidget(widget)
        self.addDockWidget(dock_area, dock)
    
    def make_dock_widgets(self):
        self.star_widget = ActiveCollectionView(
            active_collection = self.collections["stars"],
            columns=star_columns,
            selection=self.selection,
            selection_channel="star",
        )
        self.setCentralWidget(self.star_widget)
        
        self.spectra_widget = ActiveCollectionView(
            active_collection = self.collections["spectra"],
            selection=self.selection,
            columns=spectrum_columns,
            selection_channel="spectrum",
        )
        self.attach_as_dock("spectra", self.spectra_widget, Qt.BottomDockWidgetArea)
        
        self.transition_widget = ActiveCollectionView(
            active_collection = self.collections["transitions"],
            columns=base_transition_columns,
            selection=self.selection,
            selection_channel="transition",
        )
        self.attach_as_dock("transitions", self.transition_widget, Qt.RightDockWidgetArea)
        
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
    
    def get_target_collection(self):
        tcd = TargetCollectionDialog(
            collections=self.collections,
            parent=self,
        )
        tcd.exec_()
        return tcd.collection, tcd.behavior
    
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
    
    def on_new_star(self):
        new_star = NewStarDialog.get_new(parent=self)
        if not new_star is None:
            self.db.add(new_star)
    
    def on_query_db(self):
        pass
    
    def on_load(self):
        collection, behavior = self.get_target_collection()
        if collection is None:
            return

        coll_name = collection.name.lower()
        if "spect" in coll_name:
            rfunc_expr="tmb.io.read_spec"
        elif ("transition" in coll_name) or ("line" in coll_name):
            rfunc_expr = "tmb.io.read_linelist"
        else:
            rfunc_expr = ""
        ld = tmbg.loading_dialog.LoadDialog(
            rfunc_expr=rfunc_expr,
            parent=self,
        )
        ld.exec_()
        if not ld.result is None:
            if behavior == "replace":
                collection.set(ld.result)
            elif behavior == "extend":
                collection.extend(ld.result)
    
    def on_load_ll(self):
        pass
    
    def on_commit(self):
        self.db.commit()
        save_msg = "committed to {}".format(self.db.path)
        self.statusBar().showMessage(save_msg)
    
    def print_args(self, *args, **kwargs):
        """prints the arguments passed in
        useful for connecting to events during gui debugging"""
        print("in print_args")
        print(args, kwargs)

