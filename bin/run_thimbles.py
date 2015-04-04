
import os
import sys
import time

import thimblesgui as tmbg
import thimbles as tmb
from thimbles.tasks import task_registry, task
from thimbles.options import opts

@task(result_name="hello")
def hello_world(greeting="hello", subject="world!"):
    greet_str =  "{} {}".format(greeting, subject)
    print(greet_str)
    return greet_str

from thimblesgui import QtCore, QtGui, Qt
from thimblesgui import main_window
from thimbles import resource_dir
gui_resource_dir = os.path.join(os.path.dirname(tmbg.__file__),"resources")

class ThimblesMainApplication(QtGui.QApplication):
    
    def __init__ (self):
        super(ThimblesMainApplication,self).__init__([])
        self.aboutToQuit.connect(self.on_quit)
        
        ## splash screen for thimbles
        #spl_path = os.path.join(resource_dir, "splash_screen.png")
        #self.spl_pic = QPixmap(spl_path)
        #self.splash = QSplashScreen(self.spl_pic, Qt.WindowStaysOnTopHint)
        #self.splash.setMask(self.spl_pic.mask())
        #if not options.no_splash:
        #    self.splash.show()
        #    self.processEvents()
        #    time.sleep(0.01)
        #    self.processEvents()
        #    
        #    #TODO: use size to make main window the full screen size
        #    screen_rect = self.desktop().screenGeometry()
        #    size = screen_rect.width(), screen_rect.height()
        
        self.main_window = main_window.ThimblesMainWindow()
        self.main_window.show()
        #if not options.no_window:    
        #    self.main_window.show()
        
        # close the splash window
        #self.splash.finish(self.main_window)
        #if options.no_window:
        #    self.main_window.close()
    
    def on_quit (self):
        pass


if __name__ == "__main__":
    opts.parse_options()
    QtGui.QApplication.setLibraryPaths([])
    
    if not opts["no_window"]:
        try:
            app = ThimblesMainApplication()
        except RuntimeError:
            app = ThimblesMainApplication.instance()
        sys.exit(app.exec_())
