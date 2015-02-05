
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'
import matplotlib.pyplot as plt

import thimbles as tmb
from thimblesgui.mplwidget import MatplotlibWidget
from thimblesgui.spec_widget import FluxDisplay

import main_window
import dialogs
import widgets
import models
import views
import dialogs
