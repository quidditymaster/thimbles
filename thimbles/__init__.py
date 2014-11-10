"""
THIMBLES: Tools for Handling Intricate Measurements on Breathtakingly Large Ensembles of Spectra

"""
# ########################################################################### #

import dependencies

import utils
from verbosity import logger

import os
resource_dir = os.path.join(os.path.dirname(__file__), "resources")

import io
import sources
import photometry
import spectrographs
import observations
import flags
import hydrogen
import features
import profiles
import velocity
import stellar_atmospheres
import modeling
import binning
import continuum

from .spectrum import Spectrum
from .io import *

from .thimblesdb import ThimblesDB

