# the code in this directory deals with stellar atmospheres

# TODO: take the code for moog and makekurucz and include it in __path__[0]+"/core/"

#===============================================================================#
######################## USER SETUP #############################################

executables = {'moog'      :'/Applications/astro/Moog/MOOG',
               'moog07'    :'/uufs/astro.utah.edu/common/astro_data/products/moog2007-3/MOOG',
               'moogsilent':'/Applications/astro/Moog/MOOGSILENT',
               
               # LTE atmosphere code
               'makekurucz':'/Applications/astro/makekurandy/makekurucz3.e',
               'marcs'     :"/uufs/astro.utah.edu/common/astro_data1/iivans/anderton/marcs_interpolator/interpolate_marcs.py"}

######################## USER SETUP #############################################
#===============================================================================#

# Modules
from lte_atmospheres import create_lte_atmosphere, convert_atlas12_for_moog

from moog import synth as moog_synth
from moog import ewfind as moog_ewfind
from moog import run_ewfind, ewfind_model

import utils
from utils import (get_model_name, read_moog_linelist, write_moog_par, write_moog_lines_in, 
                            parse_synth_summary_out, parse_synth_standard_out, parse_abfind_summary_out,  
                            process_moog_synth_output, crop_data_table) 

from .solar_abundance_core import *

