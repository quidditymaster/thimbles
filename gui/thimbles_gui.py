
import os
import sys
import time
from PySide import QtCore,QtGui
from PySide.QtCore import *
from PySide.QtGui import *
    
_resources_dir = os.path.join(os.path.dirname(__file__),"resources")

# ########################################################################### #

class MainApplication (QApplication):
    """
    TODO: write doc string
    """
    
    def __init__ (self,options):
        super(MainApplication,self).__init__([])
        self.aboutToQuit.connect(self.on_quit)
        
        # splash screen for thimbles
        self.spl_pic = QPixmap(os.path.join(_resources_dir, "splash_screen.png"))
        self.splash = QSplashScreen(self.spl_pic, Qt.WindowStaysOnTopHint)
        self.splash.setMask(self.spl_pic.mask())
        self.splash.show()
        time.sleep(0.01)
        self.processEvents()
        
        #for _ in xrange(100):
        #    self.processEvents()
        #    time.sleep(0.001)
        
        from main_window import AppForm
        
        # TODO: use size to make main window the full screen size
        screen_rect = self.desktop().screenGeometry()
        size = screen_rect.width(), screen_rect.height()
        self.main_window = AppForm(options)
        self.main_window
        self.main_window.show()
        
        # close the splash window
        self.splash.finish(self.main_window)
    
    def on_quit (self):
        pass

def main(options):
    try:
        app = MainApplication(options)
    except RuntimeError:
        app = MainApplication.instance()
    sys.exit(app.exec_())

if __name__ == "__main__":
    import argparse
    desc = "a spectrum processing and analysis GUI"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("spectra_files", nargs="*", help="paths to one or more spectrum data files")
    parser.add_argument("-line_list", "-ll", help="the path to a linelist file to load")
    parser.add_argument("-fwidth", "-fw",  type=float, default=3.0, 
                        help="the number of angstroms on either side of the current feature to display while fitting")
    parser.add_argument("-read_func", default="read")
    parser.add_argument("-rv", type=float, default=0.0, help="optional radial velocity shift to apply")
    #parser.add_argument("-order", type=int, default=0, help="if there are multiple spectra specify which one to pull up")
    parser.add_argument("-norm", default="ones", help="how to normalize the spectra on readin options are ones and auto' ")
    parser.add_argument("-gaussian", "-g", action="store_true", help="force pure gaussian fits")
    parser.add_argument("-auto_fit", action="store_true", help="automatically do the equivalent width measurements")
    parser.add_argument("-output", "-o", default="thimbles_out.pkl", help="the name of the output file when doing automated outputs")
    #parser.add_argument("-no_window", "-nw", action="store_true", help="suppress the GUI window")
    options = parser.parse_args()
    
    main(options)
