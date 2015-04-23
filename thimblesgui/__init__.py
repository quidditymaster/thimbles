
try:
    from PySide import QtGui
    from PySide import QtCore
    which_qt = "PySide"
except ImportError:
    from PyQt4 import QtGui
    from PyQt4 import QtCore
    which_qt = "PyQt4"

Qt = QtCore.Qt

import matplotlib as mpl
mpl.use('Qt4Agg')
mpl.rcParams['backend.qt4'] = which_qt
import matplotlib.pyplot as plt

import thimbles as tmb
import os 
style_file = os.path.join(tmb.resource_dir, "matplotlibrc")
style_dict = mpl.rc_params_from_file(style_file)
mpl.rcParams.update(style_dict)
#plt.style.use(style_file) #works only in later matplotlib versions

from thimblesgui.mplwidget import MatplotlibWidget
from thimblesgui.spec_widget import FluxDisplay

from . import main_window
from . import models
from . import views
import thimblesgui.grouping_editor
