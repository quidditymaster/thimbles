#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os 
import sys 
import re 
import time
import numpy as np 
import unittest

from thimbles import StellarParameters
from thimbles.stellar_parameters import solar_parameters
from thimbles.radtran import MoogEngine
from thimbles.radtran import MarcsInterpolator

class TestRadTranEngine(unittest.TestCase):
    
    def setUp (self):
        self.wdir = os.path.dirname(__file__)
        self.ll = tmb.io.linelist_io.read_linelist(os.path.join(self.wdir, "test.ln"))
        #self.sun_spec = tmb.io.read_spec("sun.txt")
        self.sun_params = StellarParameters(5777.0, 4.44, 0.0, 0.88)
        
        
    def test_model_spectrum(self):
        spec = self.engine.model_spectrum(linelist=self.ll, stellar_params=self.sun_params, fluxing="normalized")
        #TODO: compare against a precalculated spectrum
    
    def test_model_continuum(self):
        ctm = self.engine.model_continuum(stellar_params=self.sun_params) #
        #TODO: compare against a precalculated continuum
    
    def test_ew_to_abundance (self):
        abund = self.engine.ew_to_abundance(linelist=self.ll,stellar_params=self.sun_params)
    
    def test_abund_to_ew_and_back (self):
        """ From abundances to ew and back again, a hobbit's tale by Bilbo Baggins
        """
        eng = self.engine
        ll = self.ll
        sparams = self.sun_params
        ew_col = 'ew'
        abund = eng.ew_to_abundance(linelist=ll,stellar_params=sparams,ew_col=ew_col)
        ews = eng.abundance_to_ew(linelist=ll,stellar_params=sparams,abundance=abund)
        self.assert(np.testing.assert_array_almost_equal(ews,ll[ew_col]))

class TestMoogEngine (TestRadTranEngine):
    
    def setUp (self):
        self.photosphere_engine = MarcsInterpolator(working_dir = self.wdir)
        self.engine = MoogEngine(working_dir=self.wdir)
        TestRadTranEngine.setUp(self)
    
    def test_create_multiple(self):
        with self.assertRaises(Exception):
            MoogEngine(self.wdir)

    def test_model_continuum(self):
        with self.assertRaises(NotImplementedError):
            super(TestMoogEngine, self).test_model_continuum()
    

# ########################################################################### #
if __name__ == "__main__":
    unittest.main()
    
