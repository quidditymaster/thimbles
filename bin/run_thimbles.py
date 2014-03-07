
import os
import sys
import time
from PySide import QtCore,QtGui
from PySide.QtCore import *
from PySide.QtGui import *
import thimblesgui
from thimblesgui import main_window    

_resources_dir = os.path.join(os.path.dirname(thimblesgui.__file__),"resources")

# ########################################################################### #

class MainApplication (QApplication):
    """
    TODO: write doc string
    """
    
    def __init__ (self,options):
        super(MainApplication,self).__init__([])
        self.aboutToQuit.connect(self.on_quit)
        
        # splash screen for thimbles
        spl_path = os.path.join(_resources_dir, "splash_screen.png")
        self.spl_pic = QPixmap(spl_path)
        self.splash = QSplashScreen(self.spl_pic, Qt.WindowStaysOnTopHint)
        self.splash.setMask(self.spl_pic.mask())
        self.splash.show()
        self.processEvents()
        time.sleep(0.01)
        self.processEvents()
        
        #for _ in xrange(10):
        #    self.processEvents()
        #    time.sleep(0.005)
        
        # TODO: use size to make main window the full screen size
        screen_rect = self.desktop().screenGeometry()
        size = screen_rect.width(), screen_rect.height()
        self.main_window = main_window.AppForm(options)
        self.main_window.show()
        
        # close the splash window
        self.splash.finish(self.main_window)
    
    def on_quit (self):
        pass

if __name__ == "__main__":
    from thimblesgui.options import options
    print _resources_dir
    try:
        app = MainApplication(options)
    except RuntimeError:
        app = MainApplication.instance()
    sys.exit(app.exec_())
