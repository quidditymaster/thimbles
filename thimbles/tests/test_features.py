import unittest
import numpy as np
import thimbles as tmb
from thimbles.features import *
import matplotlib.pyplot as plt
import scipy

show_diagnostic_plots = True

class TestVoigtProfileModel(unittest.TestCase):
    
    def setUp(self):
        stell_par = tmb.stellar_parameters.StellarParameters(5000.0, 3.0, 0.0, 0.25)
        self.stell_par = stell_par
        self.vpm = VoigtProfileModel(
            wv_soln=tmb.as_wavelength_solution(np.linspace(4999, 5001, 201)),
            transition=tmb.Transition(5000.0, (26, 1), 1.0, -1.0),
            stellar_parameters=stell_par,
            max_width=6.0,
            gamma=0.01,
        )
    
    def test_call(self):
        #import pdb; pdb.set_trace()
        prof_res = self.vpm()
        val_res = self.vpm.output_p.value
        #plt.plot(self.vpm.output_p.wv_sample.wvs, prof_res)
        #plt.show()
        np.testing.assert_almost_equal(prof_res, val_res)
    
    def test_teff_listen(self):
        prof_orig = self.vpm()
        teff_p = self.vpm.inputs["teff"]
        orig_teff = teff_p.value
        call_dbl_teff = self.vpm({teff_p:orig_teff*2.0})
        teff_p.value = orig_teff*2.0
        set_dbl_teff = self.vpm.output_p.value
        np.testing.assert_almost_equal(set_dbl_teff, call_dbl_teff)
        assert np.sum(np.abs(prof_orig-call_dbl_teff)) > 0.01
        if show_diagnostic_plots:
            plt.plot(prof_orig, label="teff={}".format(orig_teff))
            plt.plot(call_dbl_teff, label="teff doubled")
            plt.legend()
            plt.show()
        #import pdb; pdb.set_trace()


class TestFeatureGroupModel(unittest.TestCase):
    
    def setUp(self):
        dummy_trans = []
        min_wv = 5000.0
        max_wv = 5020.0
        npts_tot = 1000
        for i in range(10):
            twv = np.random.uniform(min_wv+1.0, max_wv-1.0)
            tep = 1.5
            trans = tmb.Transition(twv, (26, 1), tep, -1.0)
            dummy_trans.append(trans)
        transition_group = tmb.transitions.TransitionGroup(dummy_trans)
        wv_soln = tmb.as_wavelength_solution(np.linspace(min_wv, max_wv, npts_tot))
        spars = tmb.stellar_parameters.StellarParameters(5000.0, 3.0, 0.0, 1.0)
        self.fgmod = FeatureGroupModel(
            wv_soln = wv_soln, 
            transition_group=transition_group,
            stellar_parameters=spars
        )
    
    def test_call(self):
        call_res = self.fgmod()
        value_res = self.fgmod.output_p.value
        np.testing.assert_almost_equal(call_res, value_res)
    
    def test_sum_normalized(self):
        flux = self.fgmod()
        dx = scipy.gradient(self.fgmod.output_p.wv_sample.wvs)
        flux_sum = -np.sum(flux*dx)
        ntrans = len(self.fgmod.parameters)-1
        ew_p = self.fgmod.inputs["ew"]
        assert np.abs(ew_p.value*ntrans - flux_sum) < 0.01
    
    def test_teff_listen(self):
        #import pdb; pdb.set_trace()
        orig_fl = self.fgmod()
        teff_p = self.fgmod.stellar_parameters.teff_p
        orig_teff = teff_p.value
        teff_p.value = orig_teff*2.0
        set_dbl_teff = self.fgmod.output_p.value
        assert np.sum(np.abs(orig_fl-set_dbl_teff)) > 0.001
        if show_diagnostic_plots:
            plt.plot(orig_fl)
            plt.plot(set_dbl_teff)
            plt.show()


class TestGroupedFeaturesModel(unittest.TestCase):
    
    def setUp(self):
        min_wv = 5000.0
        max_wv = 5020.0
        npts_tot = 1000
        wv_soln = tmb.as_wavelength_solution(np.linspace(min_wv, max_wv, npts_tot))
        star = tmb.stellar_parameters.Star(name="test star1")
        spars = tmb.stellar_parameters.StellarParameters(5000.0, 3.0, 0.0, 1.0)
        star.stellar_parameters = spars
        dummy_tgroups = []
        for j in range(10):
            dummy_trans = []
            tep = np.random.uniform(0.0, 5.0)
            for i in range(1+j):
                twv = np.random.uniform(min_wv, max_wv)
                trans = tmb.Transition(twv, (26, 1), tep, -1.0)
                dummy_trans.append(trans)
            transition_group = tmb.transitions.TransitionGroup(dummy_trans)
            dummy_tgroups.append(transition_group)    
        grouping_standard = tmb.transitions.TransitionGroupingStandard(dummy_tgroups)
        gfm = GroupedFeaturesModel(
            star=star,
            grouping_standard=grouping_standard,
            wv_soln=wv_soln,
        )
        self.gfm = gfm
    
    def test_call(self):
        value_res = self.gfm.output_p.value
        wvs = self.gfm.output_p.wv_sample.wvs
        if show_diagnostic_plots:
            plt.plot(wvs, value_res)
            plt.show()

if __name__ == "__main__":
    unittest.main()
