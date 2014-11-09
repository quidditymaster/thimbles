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

class TestMoogEngine(unittest.TestCase):
    
    def setUp (self):
        self.wdir = os.path.dirname(___file__)
        self.photosphere_engine = MarcsInterpolator(working_dir = self.wdir)
        self.engine = MoogEngine(working_dir=self.wdir)
        self.ll = tmb.io.rea
        self.sun_params = StellarParameters(5777.0, 4.44, 0.0, 0.88)
    
    def test_create_multiple(self):
        with self.assertRaises(Exception):
            MoogEngine(self.wdir)
    
    def test_model_spectrum(self):
        spec = self.engine.model_spectrum(linelist=self.ll, stellar_params=self.sun_params, fluxing="normalized")
    
    def test_model_continuum(self):
        ctm = self.engine.model_continuum(stellar_params=self.sun_params) #

    def test_ew_to_abundance (self):
        abund = self.engine.ew_to_abundance(linelist=self.ll,stellar_params=self.sun_params)
        
    def test_ews (self):
        abund = self.engine.ew_to_abundance(linelist=self.ll,stellar_params=self.sun_params)
        ll = self.engine.abundance_to_ew(linelist=self.ll,stellar_params=self.sun_params)

        self.assert(np.testing.assert_array_almost_equal(ll['ew'],self.ll['ew'])



# ########################################################################### #
if __name__ == "__main__":
    unittest.main()
    
