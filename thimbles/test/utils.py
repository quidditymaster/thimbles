#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Purpose: Utilities for Thimbles
# Author: Dylan Gregersen
# Date: Jan 18, 2014

# ########################################################################### #

# Standard Library
from __future__ import division, print_function, absolute_import
from collections import OrderedDict
# 3rd Party
import numpy as np
import h5py

# Internal
from ..stellar_atmospheres import solar_abundance as abund_standard
from ..utils  import piecewise_polynomial

from .. import __path__ as timbles_path
tpath = timbles_path[0]

# ########################################################################### #
cog_ppol_hf = h5py.File("%s/resources/cog_ppol.h5" % tpath)

_elements_params = {'La I':dict(n=10),
                    'Fe I':dict(n=320),
                    'O III':dict(n=1),
                    'Au I':dict(n=1),
                    'Zn I':dict(n=10),
                    'Na I':dict(n=4),
                    'Ca II':dict(n=50),
                    'Ti I':dict(n=250),
                    'Y II':dict(n=20),
                    'U I':dict(n=2),
                    'Hg I':dict(n=1),
                    'Yb I':dict(n=1),
                    'Fe II':dict(n=130),
                    'Ca I':dict(n=110),
                    'Cr II':dict(n=60),
                    'Pb II':dict(n=2),
                    'Zr I':dict(n=10),
                    'Ag I':dict(n=10),
                    'Eu I':dict(n=1),
                    'V I':dict(n=10),
                    'Mg I':dict(n=60),
                    'Sc II':dict(n=20),
                    'Nb II':dict(n=10),
                    'Ti II':dict(n=90),
                    'Cu I':dict(n=60),
                    'Ba II':dict(n=15),
                    'Ni I':dict(n=90),
                    'Mg II':dict(n=20),
                    'Os II':dict(n=2),
                    'Co I':dict(n=70),
                    'Mn I':dict(n=30),
                    'Si I':dict(n=10),
                    'Cr I':dict(n=20)}



def estimate_lorentz_width (x,iqp_deriv):
    """
    
    """
    lorz_width = lambda xval: 0.5*(1-iqp_deriv(xval)**2)
    # TODO: make this better, sqrt part of COG is bad
    return lorz_width(x)

def generate_random_linelist (teff,wv_bounds=(4500,5500),params=None,filepath=None):
    """
    Randomly sample wavelengths 
    
    Parameters
    ----------
    params : dict
        keys are the species id (e.g. "Fe I") and the values are lists of 
        parameters [number_lines]
    
    """
    abund_offset_range = (-1,1)
    species_offset_range = (-1,1)
    ep_range = (0,12)
    loggf_range = (-6.0,0.5) 
    
    theta = 5040.0/teff
    
    #     # TODO: remove this calculation???
    #     #     # fix to a particular line which should be by the turnoff
    #     #     # Fe I    88.2 2.22 EP -4.2 loggf
    #     loggf = -4.2
    #     ep = 2.22
    #     x_turnoff = abund_standard['Fe']['abundance']+loggf-theta*ep
    #     x-x_turnoff  = -5
    #         
    # based on the model abundance used in the cog file
    xnorm = -6.5
    ynorm = -2.0
    
    # read in the parameters 
    if params is None:
        params = _elements_params
    el_params = params.copy()
    for el,pars in _elements_params.iteritems():
        el_params.setdefault(el,pars)
    

    coeffs, knots, centers, scales = np.array(cog_ppol_hf["coefficients"]), np.array(cog_ppol_hf["knots"]), np.array(cog_ppol_hf["centers"]), np.array(cog_ppol_hf["scales"])
    iqp = piecewise_polynomial.InvertiblePiecewiseQuadratic(coeffs, knots, centers=centers, scales=scales)
    iqp_deriv = iqp.deriv()
    
    # calc the linelist
    linelist = {}
    element_abund = {}
    for species,pars in params.items():
        wvs = np.random.uniform(wv_bounds[0],wv_bounds[1],pars['n'])
        solar_abund_offset = np.random.uniform(*abund_offset_range)
        
        # get the abundance for this element, ignore species
        abund = abund_standard[species]['abundance']+solar_abund_offset
        element_abund.setdefault(abund_standard[species]['element'],abund) 
    
        species_offset = np.random.uniform(*species_offset_range)    
        species_abund = element_abund[abund_standard[species]['element']]+species_offset
        species_abund = np.repeat(species_abund,pars['n'])
        
        # generate the parameters for the lines
        spe_col = np.repeat(abund_standard.species_id(species),pars['n'])
        ep = np.random.uniform(ep_range[0],ep_range[1],pars['n'])
        loggf = np.random.uniform(loggf_range[0],loggf_range[1],pars['n'])
        
        # calculate the line strengths from the COG
        x = species_abund + loggf - theta*ep + xnorm
        logrw = iqp(x)+ynorm
        ew = (10**logrw)/wvs*1000.0 # mA        

        # estimate the lorzentian and gaussian widths for this line
        lorz_width = estimate_lorentz_width(x, iqp_deriv)
        gauss_width = np.repeat(99.9,pars['n'])
    
        # add to the linelist
        linelist[species] = np.dstack((wvs,spe_col,ep,loggf,ew,gauss_width,lorz_width))[0]
                             
    if filepath is not None:
        # save moog file
        f = open(filepath,'w')
        header = "# Fake linelist created THIMBLES with teff {} # "
        header += "wvs species ep loggf ew gauss_width lorz_width # "
        header += "guassian and lorentzian widths are estimate\n"
        f.write(header.format(teff))
        
        fmt = "{0:>9.5f} {1:>9.1f} {2:>9.2f} {3:>9.2f}"+20*" "+" {4:>9.2f}"+10*" "
        fmt += " {5:>9.2f} {6:>9.2f} FAKE_LINE\n"
        for species,ll in linelist.iteritems():
            for row in ll:
                f.write(fmt.format(*row))    
    return linelist
        
        

    
    
    
    
    

