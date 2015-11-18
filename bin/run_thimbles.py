
import sys

#import thimblesgui as tmbg
from thimblesgui.application import ThimblesMainApplication
from thimbles.options import opts
from thimblesgui import QtCore, QtGui, Qt

if __name__ == "__main__":
    import sys
    opts.parse_commands(sys.argv[1:])
    #QtGui.QApplication.setLibraryPaths([])
    
    try:
        app = ThimblesMainApplication()
    except RuntimeError:
        app = ThimblesMainApplication.instance()
    sys.exit(app.exec_())
