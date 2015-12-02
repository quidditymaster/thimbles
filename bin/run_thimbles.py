
import sys

#import thimblesgui as tmbg
from thimblesgui.application import ThimblesMainApplication
from thimblesgui.main_window import ThimblesMainWindow
from thimbles.options import opts
from thimblesgui import QtCore, QtGui, Qt

#QtGui.QApplication.setLibraryPaths([])    
try:
    app = ThimblesMainApplication()
except RuntimeError:
    app = ThimblesMainApplication.instance()
main_window = ThimblesMainWindow(app)

main_window.show()
sys.exit(app.exec_())
