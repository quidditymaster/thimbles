
import os
import sys
import time

import thimblesgui as tmbg
import thimbles as tmb

from thimbles.options import opts

from thimblesgui.main_window import ThimblesMainWindow
from thimblesgui import QtCore, QtGui, Qt
gui_resource_dir = os.path.join(os.path.dirname(tmbg.__file__),"resources")

class ThimblesMainApplication(QtGui.QApplication):
    
    def __init__ (self):
        super(ThimblesMainApplication,self).__init__([])
        self.aboutToQuit.connect(self.on_quit)
        
        ## splash screen for thimbles
        self.splash = None
        if opts["GUI.show_splash"]:
            spl_path = os.path.join(gui_resource_dir, "splash_screen.png")
            self.spl_pic = QtGui.QPixmap(spl_path)
            self.splash = QtGui.QSplashScreen(self.spl_pic, Qt.WindowStaysOnTopHint)
            self.splash.setMask(self.spl_pic.mask())
            self.splash.show()
            self.processEvents()
            for i in range(10):
                time.sleep(0.01)
                self.processEvents()
        
        db_path = tmb.opts["GUI.project_path"]
        self.project_db = tmb.ThimblesDB(db_path)
        
        #self.main_window = ThimblesMainWindow(db_path)
        #if not self.splash is None:
        #    self.splash.finish(self.main_window)
        #self.main_window.show()
        #screen_rect = self.desktop().screenGeometry()
        #print(screen_rect)
        #size = screen_rect.width(), screen_rect.height()
        #self.main_window.show()
    
    def finish_splash(self, mw):
        if not self.splash is None:
            self.splash.finish(mw)
    
    def on_quit (self):
        pass
