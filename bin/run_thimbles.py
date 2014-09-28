
import os
import sys
import time

from thimbles.tasks import task_registry, task_parser, task

#from PySide import QtCore,QtGui
#from PySide.QtCore import *
#from PySide.QtGui import *
#import thimblesgui as tmbg
#from thimblesgui import main_window
#from thimblesgui import workers

#print "main window file", main_window.__file__
#_resources_dir = os.path.join(os.path.dirname(tmbg.__file__),"resources")

# ########################################################################### #

#class MainApplication (QApplication):
#    
#    def __init__ (self,options):
#        super(MainApplication,self).__init__([])
#        self.aboutToQuit.connect(self.on_quit)
#        
#        # splash screen for thimbles
#        spl_path = os.path.join(_resources_dir, "splash_screen.png")
#        self.spl_pic = QPixmap(spl_path)
#        self.splash = QSplashScreen(self.spl_pic, Qt.WindowStaysOnTopHint)
#        self.splash.setMask(self.spl_pic.mask())
#        if not options.no_splash:
#            self.splash.show()
#            self.processEvents()
#            time.sleep(0.01)
#            self.processEvents()
#            
#            #TODO: use size to make main window the full screen size
#            screen_rect = self.desktop().screenGeometry()
#            size = screen_rect.width(), screen_rect.height()
#        self.main_window = main_window.AppForm(options)
#        if not options.no_window:    
#            self.main_window.show()
#        
#        # close the splash window
#        self.splash.finish(self.main_window)
#        if options.no_window:
#            self.main_window.close()
#    
#    def on_quit (self):
#        pass


@task()
def print_success():
    print "hello, hooray!"
    return 2

from thimbles.tasks import task

if __name__ == "__main__":
    arg_list = sys.argv[1:]
    
    task_name = arg_list.pop(0)
    cur_task = task_registry.get(task_name)
    if not task is None:
        cur_task.run_task()
    
    #from thimblesgui.options import options
    #try:
    #    app = MainApplication(options)
    #except RuntimeError:
    #    app = MainApplication.instance()
    #if not options.no_window:
    #    sys.exit(app.exec_())
