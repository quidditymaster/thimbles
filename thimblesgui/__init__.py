
import matplotlib as mpl
mpl.use('Qt4Agg')
mpl.rcParams['backend.qt4'] = 'PySide'
import matplotlib.pyplot as plt

import thimbles as tmb
import os 
plt.style.use(os.path.join(tmb.resource_dir, "matplotlibrc"))

from thimblesgui.mplwidget import MatplotlibWidget
from thimblesgui.spec_widget import FluxDisplay

import main_window
import dialogs
import widgets
import models
import views
import dialogs
import thimblesgui.grouping_editor
