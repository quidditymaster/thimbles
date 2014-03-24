import os

import numpy as np
import h5py

import thimbles as tmb

resources_dir = os.path.join(os.path.dirname(tmb.__file__), "resources")
standards_file = os.path.join(resources_dir, "standards.h5")


class StandardsDatabase(object):
    
    def __init__(self, standards_file=standards_file):
        pass
    
    def extract_standard(self):
        pass
    
    def inject_standard(self):
        pass
    
    def match_standard(self, spectra):
        wv_bounds = [(s.wv[0], s.wv[-1]) for s in spectra]
        min_wv = np.min(wv_bounds)
        max_wv = np.max(wv_bounds)
        