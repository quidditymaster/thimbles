
import utils
from verbosity import Verbosity
verbosity = Verbosity(level='silent')

import os
resource_dir = os.path.join(os.path.dirname(__file__), "resources")

from thimbles import io
from thimbles import sources
from thimbles import photometry
from thimbles import spectrographs
from thimbles import observations
from thimbles import flags
from thimbles import features
from thimbles import profiles
from thimbles import velocity
from thimbles import stellar_atmospheres
from thimbles import modeling
from thimbles import binning
from thimbles import hydrogen

from .spectrum import Spectrum
from .io import *

from thimbles.thimblesdb import ThimblesDB

