#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os 
import sys 
import re 
import time
import numpy as np 
import unittest

import thimbles as tmb
from thimbles.stellar_parameters import StellarParameters
from thimbles.radtran import MoogEngine
from thimbles.radtran import MarcsInterpolator

#_make_plots = True
_make_plots = False

class TestRadTranEngine(unittest.TestCase):
    
    def setUp (self):
        self.wdir = os.path.dirname(__file__)
        self.ll = tmb.io.linelist_io.read_linelist(os.path.join(self.wdir, "test.ln"))
        #self.photosphere_engine = MarcsInterpolator(working_dir = self.wdir)
        #self.sun_spec = tmb.io.read_spec("sun.txt")
        self.sun_params = StellarParameters(5777.0, 4.44, 0.0, 0.88)
        self.sun_mod_file = os.path.join(self.wdir, "sun.mod")
        self.engine = MoogEngine(working_dir = self.wdir)
    
    def test_model_spectrum(self):
        spec = self.engine.spectrum(linelist=self.ll, stellar_params=self.sun_mod_file, wavelengths=np.linspace(15005, 150020, 500), normalized=True)
        if _make_plots:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(spec.wvs, spec.flux)
            ax.set_xlabel("Wavelength")
            ax.set_ylabel("Flux")
            plt.show()
        assert np.max(spec.flux) > 0.99
        min_flux = np.min(spec.flux)
        assert min_flux < 0.99
        assert min_flux > 0.0
    
    #def test_model_continuum(self):
    #    ctm = self.engine.continuum(stellar_params=self.sun_mod_file) #
    #    #TODO: compare against a precalculated continuum
    
    #def test_abundance_to_ew(self):
    #    abres = self.engine.abundance_to_ew(self.ll, stellar_params=self.sun_mod_file)
    #    import pdb; pdb.set_trace()
    
    def test_line_abundance (self):
        llabs = self.engine.line_abundance(linelist=self.ll,stellar_params=self.sun_mod_file)
        #import pdb; pdb.set_trace()
    
    #def test_abund_to_ew_and_back (self):
    #    """ From abundances to ew and back again, a hobbit's tale by Bilbo Baggins
    #    """
    #    eng = self.engine
    #    ll = self.ll
    #    sparams = self.sun_mod_file
    #    ew_col = 'ew'
    #    abund = eng.ew_to_abundance(linelist=ll,stellar_params=sparams,ew_col=ew_col)
    #    ews = eng.abundance_to_ew(linelist=ll,stellar_params=sparams,abundance=abund)
    #    self.assertTrue(np.testing.assert_array_almost_equal(ews,ll[ew_col]))
    #
    ##def test_create_multiple(self):
    ##    with self.assertRaises(Exception):
    ##        MoogEngine(self.wdir)
    #    

# ########################################################################### #
if __name__ == "__main__":
    unittest.main()
    
