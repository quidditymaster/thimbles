from __future__ import print_function
import time
import threading

import numpy as np
import scipy

import thimblesgui as tmbg

class Worker(threading.Thread):
    
    def __init__(self, work_queue):
        threading.Thread.__init__(self)
        self.work_queue = work_queue
        wname = "thimbles worker @{}".format(id(self))
        self.setName(wname)
        self._stop = threading.Event()
    
    def cleanup(self):
        pass
    
    def stop(self):
        try:
            self.cleanup()
        except Exception as e:
            print(e)
        self._stop.set()
    
    def stopped(self):
        return self._stop.isSet()
