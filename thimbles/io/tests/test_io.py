
# Standard Library
import unittest
import os

# 3rd Party
import numpy as np
from astropy.io import fits

# Internal
import thimbles
# from thimbles.io import (read,
#                          read_many_fits)
# # read_txt, 
# # ExtractWavelengthCoefficients, 
from thimbles.utils.misc import var_2_inv_var

resources_path = os.path.join(os.path.dirname(thimbles.__file__),"resources/test_data")

class DummyTest(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_auto_passes(self):
        pass

# ########################################################################### #

# class TestWavelengthFunctions (unittest.TestCase):
    
#     def setUp(self):
#         unittest.TestCase.setUp(self)

# class TestExtractWavlengthCoefficients (unittest.TestCase):
#     def setUp(self):
#         unittest.TestCase.setUp(self)
#     def test_hd221170_read (self):
#         filename = "hd221170_3500sm_header.fits"
#         hdu = fits.open(os.path.join(resources_path,filename))[0]
#         ewc = ExtractWavelengthCoefficients(hdu.header)
#           
#         spectre_history = ewc.get_SPECTRE_history()
#        
#         self.assert_(len(spectre_history)==1, "Found too many different tags")
#        
#         key = spectre_history.keys()[0]
#         self.assert_(key==1121065200.0, "Found wrong time stamp")
#        
#         _,func_type,coeffs = spectre_history[key]
#         self.assertEqual(func_type,'chebyshev poly', "wrong spectre function type")
#        
#         solution = np.array([3497.88543141, 30.9881846801, -0.666763321767, -0.0528877896848, 0.000482608059658, 0.0])
#         self.assert_(np.all(np.array(coeffs)==solution),"Wrong coefficients found")

# class TestReadFunctions (unittest.TestCase):
    
#     def setUp(self):
#         unittest.TestCase.setUp(self)
    
#     def test_read_txt (self):
#         filepath = os.path.join(resources_path,"text_data.txt")

#         sol_wvs = np.array([ 2500.03711,  2500.07446,  2500.11206,  2500.14941,  2500.18677,
#         2500.22437,  2500.26172,  2500.29907,  2500.33667,  2500.37402])        
        
#         sol_flux = np.array([ 1.02655196,  1.0275799 ,  1.01439083,  1.01318395,  1.01666176,
#         1.01223826,  1.00718141,  1.00520563,  0.99980551,  0.99655437])        
        
#         sol_inv_var = np.array([ 194275.23827509,  194080.89495597,  196604.32718104,
#         196838.51745852,  196165.16965363,  197022.41508907,
#         198011.62397424,  198400.82534234,  199472.42202212,
#         200123.17604924])
        
#         spec = read_txt(filepath)[0]
        
#         wv = spec.wv
#         flux = spec.flux
#         inv_var = spec.inv_var
        
#         within_error = lambda first,second: np.all(np.abs(first-second)<1e-08)
        
#         self.assert_(within_error(wv,sol_wvs),"extracted wrong wavelengths from text")
#         self.assert_(within_error(flux,sol_flux),"extracted wrong flux points")
#         self.assert_(within_error(inv_var,sol_inv_var),"wrong place values for inverse variance")
        
#     def test_hst_bintable (self):
        
#         filepath = os.path.join(resources_path,"hst_bintable_example.fits")
#         spectra = read(filepath)
        
#         self.assertEqual(len(spectra),3, "wrong number of orders read")
        
#         hdulist = fits.open(filepath)
#         hdu1 = hdulist[1]
        
#         for i in xrange(3):
            
#             self.assertEqual(hdulist[0].header,spectra[i].metadata['header0'],"primary headers not equal")
#             self.assertEqual(hdulist[1].header,spectra[i].metadata['header1'],"secondary headers not equal")
            
            
#             self.assert_(np.all(hdu1.data['WAVELENGTH'][i]==spectra[i].wv),
#                          "wavelengths are not equal for hst data")

#             self.assert_(np.all(hdu1.data['FLUX'][i]==spectra[i].flux),
#                          "flux are not equal for hst data")
        
        
#             inv_var = var_2_inv_var(hdu1.data['ERROR'][i]**2)
#             self.assert_(np.all(inv_var==spectra[i].inv_var),
#                          "inverse variance are not equal for hst data")      

#     def test_spectre_save (self):
#         filepath = os.path.join(resources_path,"spectre_save.fits")
#         spectre_answer = np.loadtxt(os.path.join(resources_path,"spectre_save.txt"))
        
#         spec = read(filepath)[0]
        
#         wvequal = np.all(np.abs(spec.wv-spectre_answer[:,0])<0.000249)
#         fluxequal = np.all(np.abs(spec.flux-spectre_answer[:,1]<5.3e-9))
        
#         self.assert_(wvequal,"wavelengths not equal to SPECTRE")
#         self.assert_(fluxequal,"flux values equal to SPECTRE")
    
#     def test_read_many_fits (self):
#         filepath = os.path.join(resources_path,"spectre_save.fits")
#         nspec = 3
#         filelist = [filepath for _ in xrange(nspec)]
#         spectra = read_many_fits(filelist)
        
#         self.assertEqual(len(spectra), nspec, "wrong number of output spectra")
                

if __name__ == "__main__":
    unittest.main()
