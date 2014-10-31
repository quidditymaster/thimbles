
import os
import sys
import time

import thimbles as tmb
from thimbles.tasks import task_registry, task
from thimbles.options import opts

@task()
def hello_world(greeting="hello", subject="world"):
    greet_str =  "{} {}".format(greeting, subject)
    print greet_str
    return greet_str

@task()
def annother_task(a, b, c=3.0):
    print a+b+c
    return a+b+c

opts.parse_options()

from PySide import QtCore,QtGui
from PySide.QtCore import *
from PySide.QtGui import *
import thimblesgui as tmbg
from thimblesgui import main_window

print "main window file", main_window.__file__
_resources_dir = os.path.join(os.path.dirname(tmbg.__file__),"resources")

class MainApplication (QApplication):
    
    def __init__ (self):
        super(MainApplication,self).__init__([])
        self.aboutToQuit.connect(self.on_quit)
        
        ## splash screen for thimbles
        #spl_path = os.path.join(_resources_dir, "splash_screen.png")
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
        
        self.main_window = main_window.AppForm()
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
    try:
        app = MainApplication()
    except RuntimeError:
        app = MainApplication.instance()
    sys.exit(app.exec_())

