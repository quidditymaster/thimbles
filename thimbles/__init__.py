thimbles_header_str =\
"""
THIMBLES:
  Tools for            ##########       |  
  Handling            ############      |  
  Integrated         ##############     |  
  Modeling of      _ ##############     |  
  Breathtakingly _/ \############## __  |   
  Large         /    ##############/   \|  
  Ensembles of /    ################   (\)  ____
  Spectra  ___/    ##################    \_/    \_
"""
__doc__ = thimbles_header_str

_with_numba = False
#try:
#    import numba
#    _with_numba = True
#except ImportError:
#    _with_numba = False

import os
resource_dir = os.path.join(os.path.dirname(__file__), "resources")
speed_of_light = 299792.458 #speed of light in km/s
from . import options 

from . import pseudonorms
from . import thimblesdb
from . import modeling
from .periodictable import ptable, atomic_number, atomic_symbol
from . import io
from . import sources
from .star import Star
from . import photometry
from . import spectrographs
from . import observations
from . import flags
from . import hydrogen
from . import profiles
from . import coordinatization
from .spectrum import Spectrum
from . import features
from . import continuum
from . import coadd
from . import noise
from . import radtran
from . import resampling
from . import transitions
from . import charts
from . import velocity
from .radtran import mooger
from . import rotation
from . import tellurics
from . import cog

from .thimblesdb import ThimblesDB
from . import analysis

from thimbles import workingdataspace as wds

opts = options.opts
from . import config
opts.run_config()

wds.__dict__.update(dict(thimblesdb.Base._decl_class_registry))
