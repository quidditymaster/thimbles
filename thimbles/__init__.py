thimbles_header_str =\
"""
THIMBLES:
  Tools for            ##########       |  
  Handling            ############      |  
  Intricate          ##############     |  
  Measurements on  _ ##############     |  
  Breathtakingly _/ \############## __  |   
  Large         /    ##############/   \|  
  Ensembles of /    ################   (\)  ____
  Spectra  ___/    ##################    \_/    \_
"""
__doc__ = thimbles_header_str

try:
    import numba
    _with_numba = True
except ImportError:
    _with_numba = False

import os
resource_dir = os.path.join(os.path.dirname(__file__), "resources")

from thimbles import dependencies
from thimbles.verbosity import logger
from thimbles import modeling
from thimbles.periodictable import ptable
from thimbles import io
from thimbles.io import *
from thimbles import sources
from thimbles import photometry
from thimbles import spectrographs
from thimbles import observations
from thimbles import flags
from thimbles import hydrogen
from thimbles import profiles
from thimbles import coordinatization
from thimbles.spectrum import Spectrum, as_wavelength_solution
from thimbles import features
#from thimbles import velocity
from thimbles import stellar_atmospheres
from thimbles import continuum
from thimbles import radtran
from thimbles import resampling

from thimbles.thimblesdb import ThimblesDB
