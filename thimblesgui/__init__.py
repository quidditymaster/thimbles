
import os 

import matplotlib as mpl
try:
    from PyQt4 import QtGui
    from PyQt4 import QtCore
    mpl.use("Qt4Agg")
    which_qt = "PyQt4"
except ImportError:
    from PyQt5 import QtWidgets
    from PyQt5 import QtGui
    from PyQt5 import QtCore
    mpl.use("Qt5Agg")
    which_qt = "PyQt5"

#mpl.rcParams['backend.qt4'] = which_qt
Qt = QtCore.Qt

#import thimbles as tmb
#style_file = os.path.join(tmb.resource_dir, "matplotlibrc")
#style_dict = mpl.rc_params_from_file(style_file)
#mpl.rcParams.update(style_dict)
#plt.style.use(style_file) #works only in later matplotlib versions

from . import selection
from . import active_collections
from thimblesgui.mplwidget import MatplotlibWidget
from thimblesgui.spec_widget import FluxDisplay

#from . import main_window
#from . import models
#from . import views
#import thimblesgui.grouping_editor
