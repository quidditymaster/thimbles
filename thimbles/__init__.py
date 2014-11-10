thimbles_header_str =\
"""
THIMBLES:
  Tools for            ##########
  Handling            ############
  Intricate          ##############
  Measurements on    ############## 
  Breathtakingly     ##############
  Large              ##############
  Ensembles of      ################
  Spectra          ##################
"""

__doc__ = """
THIMBLES: Tools for Handling Intricate Measurements on Breathtakingly Large Ensembles of Spectra

"""+thimbles_header_str

import dependencies
import utils
from verbosity import logger

import os
resource_dir = os.path.join(os.path.dirname(__file__), "resources")

from thimbles import io
from thimbles import sources
from thimbles import photometry
from thimbles import spectrographs
from thimbles import observations
from thimbles import flags
from thimbles import hydrogen
from thimbles import features
from thimbles import profiles
from thimbles import velocity
from thimbles import stellar_atmospheres
from thimbles import modeling
from thimbles import binning
from thimbles import continuum

from thimbles.spectrum import Spectrum
from thimbles.io import *

from thimbles.thimblesdb import ThimblesDB

