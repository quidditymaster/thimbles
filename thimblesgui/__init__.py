
import matplotlib as mpl
mpl.use('Qt4Agg')
mpl.rcParams['backend.qt4'] = 'PySide'
import matplotlib.pyplot as plt

import thimbles as tmb
import os 
style_file = os.path.join(tmb.resource_dir, "matplotlibrc")
style_dict = mpl.rc_params_from_file(style_file)
mpl.rcParams.update(style_dict)
#plt.style.use(style_file) #works only in later matplotlib versions

from thimblesgui.mplwidget import MatplotlibWidget
from thimblesgui.spec_widget import FluxDisplay

import main_window
import models
import views
import thimblesgui.grouping_editor
