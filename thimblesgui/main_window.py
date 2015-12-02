
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
from thimbles.contexts import contextualizers

from thimbles import ThimblesDB
gui_resource_dir = os.path.join(os.path.dirname(__file__),"resources")

# ########################################################################### #


class ThimblesMainWindow(QtGui.QMainWindow):
    
    def __init__(self, app):
        super(ThimblesMainWindow, self).__init__()
        self.db = app.project_db#ThimblesDB(project_db_path)
        app.finish_splash(self)
        
        self.setWindowTitle("Thimbles")
        self.make_collections()
        self.selection = tmbg.selection.GlobalSelection(
            channels=[
            "source",
            "spectrum",
            "transition",
        ],)
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
        
        #self.modeling_pattern = tmb.analysis.ModelingPattern(
        #    model_templates = tmb.analysis.model_component_recipes["spectral default"],
        #    context_extractors = {
        #        "global": lambda x : self.global_params,
        #        "spectrum": lambda x: x,
        #        "exposure": lambda x: x.exposure,
        #        "order":lambda x: x.order,
        #        "chip": lambda x: x.chip,
        #        "aperture": lambda x: x.aperture,
        #    },
        #)
        
        self.make_actions()
        self.make_menus()
        self.make_tool_bar()
        self.make_status_bar()
        self.make_dock_widgets()
    
    def make_collections(self):
        self.collections = {}
        
        spectra_collection = ActiveCollection(
            name="spectra",
            db=self.db,
            default_columns=spectrum_columns,
            default_query="db.query(Spectrum)",
            default_read_func="tmb.io.read_spec",
        )
        self.collections["spectra"] = spectra_collection
        
        star_collection = ActiveCollection(
            name="stars",
            db=self.db,
            default_columns=star_columns,
            default_read_func="tmb.io.read_spec",
            default_query="db.query(Star).offset(0).limit(50)",
        )
        self.collections["stars"] = star_collection

        default_tq = "db.query(Transition).join(Ion).filter(Transition.wv > 2000).filter(Transition.wv < 20000).offset(0).limit(1000)"
        transition_collection = ActiveCollection(
            name="transitions",
            db=self.db,
            default_columns=base_transition_columns,
            default_read_func="tmb.io.read_linelist",
            default_query=default_tq,
        )
        self.collections["transitions"] = transition_collection
    
    def make_actions(self):
        #QtGui.QAction(QtGui.QIcon(":/images/new.png"), "&Attach Database", self)
        self.apply_mct_act = QtGui.QAction("apply MCT", self)
        self.apply_mct_act.setStatusTip("generate component models for an appropriate set of contexts")
        self.apply_mct_act.triggered.connect(self.on_apply_mct)
        
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
        self.file_menu.addAction(self.save_act)
        self.file_menu.addAction(self.quit_act)
        
        self.context_menu = menu_bar.addMenu("&Context")
        source_menu = self.context_menu.addMenu("Sources")
        source_menu.addAction(self.new_star_act)
        
        self.modeling_menu = menu_bar.addMenu("&Modeling")
        self.modeling_menu.addAction(self.apply_mct_act)
        
        self.help_menu = self.menuBar().addMenu("&Help")
        self.help_menu.addAction(self.about_act)
    
    def make_tool_bar(self):
        self.file_tool_bar = self.addToolBar("File")
        self.file_tool_bar.addAction(self.save_act)
    
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
            selection=self.selection,
            selection_channel="source",
        )
        self.setCentralWidget(self.star_widget)
        
        self.spectra_widget = ActiveCollectionView(
            active_collection = self.collections["spectra"],
            selection=self.selection,
            selection_channel="spectrum",
        )
        self.attach_as_dock("spectra", self.spectra_widget, Qt.BottomDockWidgetArea)
        
        self.transition_widget = ActiveCollectionView(
            active_collection = self.collections["transitions"],
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
    
    def on_apply_mct(self):
        mctad = MCTApplicationDialog(parent=self)
        mctad.exec_()
        #spectra = self.collections["spectra"]
        #self.modeling_pattern.incorporate_data(spectra)
    
    def on_new_star(self):
        new_star = NewStarDialog.get_new(parent=self)
        if not new_star is None:
            self.db.add(new_star)
    
    def on_save(self):
        self.db.commit()
        save_msg = "committed to {}".format(self.db.path)
        self.statusBar().showMessage(save_msg)
    
    def print_args(self, *args, **kwargs):
        """prints the arguments passed in
        useful for connecting to events during gui debugging"""
        print("in print_args")
        print(args, kwargs)

