import time
import threading

import numpy as np
import scipy

import thimblesgui as tmbg

class Worker(threading.Thread):
    
    def __init__(self, work_queue):
        threading.Thread.__init__(self)
        self.work_queue = work_queue
        self.setName("thimbles worker" + "@%d" % id(self))
        self._stop = threading.Event()
    
    def cleanup(self):
        pass
    
    def stop(self):
        self.cleanup()
        self._stop.set()
    
    def stopped(self):
        return self._stop.isSet()