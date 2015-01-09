import numpy as np
import pandas as pd
import thimbles as tmb
import matplotlib.pyplot as plt
from copy import copy
import scipy
import scipy.integrate as integrate
import scipy.sparse
from flags import FeatureFlags
from thimbles import ptable
from thimbles.profiles import voigt
from thimbles import resource_dir
from thimbles.modeling import Model
from thimbles.utils.misc import smooth_ppol_fit
import thimbles.utils.piecewise_polynomial as ppol
from thimbles import logger
from thimbles import hydrogen
from latbin import matching
from thimbles import as_wavelength_solution
from thimbles.utils.misc import saturated_voigt_cog
from thimbles.sqlaimports import *

def indicator_factory(indval, tolerance):
    return lambda x: np.abs(x-indval) < tolerance

class SpeciesGrouper(object):
    
    def __init__(self, group_indicators, ungrouped_val="unique", tolerance=1e-5):
        """species grouper object
        
        inputs
        
        group_indicators: list of functions
          if a function in group_indicators returns true for a species value
          that species will be given a group_num which is the index of that 
          function in the list.
          if more than one function in the list would return true for a given
          species the first function in the list takes precedence.
          The functions are called once for each possible species input
          and the result is cached so one need not worry about optimizing them.
        ungrouped_val: string
          "last" species not appearing in one of the group_indicators lists
            will be grouped together with group_num == len(group_indicators) -1
          "unique" species not appearing in one of the passed in lists will
            be grouped individually, with group num dictated by the first
            time species_group_num is called with an example of that species.
          "raise" if there is no match raise a ValueException
        tolerance: float
          the max allowed difference to accept a species as a match when
          generating extra indicators to handle grouping of species values
          for which no group_indicator function returns True.
        """
        self.group_indicators = group_indicators
        ungr_opts = ["last", "unique", "raise"]
        if ungrouped_val not in ungr_opts:
            raise ValueError("ungrouped_val type {} not recognized must be one of {}".format(ungrouped_val, ungr_opts))
        self.ungrouped_val = ungrouped_val
        self.tolerance = tolerance
    
    def __call__(self, species):
        #import pdb; pdb.set_trace()
        species = np.atleast_1d(species)
        unique_species, recon_idxs = np.unique(species, return_inverse=True)
        grp_nums = np.zeros(unique_species.shape)
        for input_sp_idx in range(len(unique_species)):
            input_sp_val = unique_species[input_sp_idx]
            grp_num_found = False
            for sp_grp_num in range(len(self.group_indicators)):
                if self.group_indicators[sp_grp_num](input_sp_val):
                    cur_group_num = sp_grp_num
                    grp_num_found = True
                    break
            if not grp_num_found:
                if self.ungrouped_val == "raise":
                    raise ValueError("given species {} was not accepted by any indicator function".format(input_sp_val))
                elif self.ungrouped_val == "unique":
                    self.group_indicators.append(indicator_factory(input_sp_val, self.tolerance))
                cur_group_num = len(self.group_indicators)-1
            grp_nums[input_sp_idx] = cur_group_num
        return grp_nums[recon_idxs]


def voigt_feature_matrix(wv_soln, centers, sigmas, gammas=None):
    wv_soln = as_wavelength_solution(wv_soln)
    indexes = []
    profiles = []
    n_features = len(centers)
    if gammas is None:
        gammas = np.zeros(n_features)
    assert len(sigmas) == n_features
    assert len(gammas) == n_features
    window_deltas = 35.0*gammas
    alt_wid = 5.0*np.sqrt(sigmas**2 + gammas**2)
    window_deltas = np.where(window_deltas > alt_wid, window_deltas, alt_wid)
    wavelengths = wv_soln.get_wvs()
    for col_idx in range(n_features):
        ccent = centers[col_idx]
        csig = sigmas[col_idx]
        cgam = gammas[col_idx]
        cdelt = window_deltas[col_idx]
        lb, ub = wv_soln.get_index([ccent-cdelt, ccent+cdelt], clip=True, snap=True)
        prof = voigt(wavelengths[lb:ub+1], ccent, csig, cgam)
        indexes.append(np.array([np.arange(lb, ub+1), np.repeat(col_idx, len(prof))]))
        profiles.append(prof)
    indexes = np.hstack(indexes)
    profiles = np.hstack(profiles)
    npts = len(wavelengths)
    mat = scipy.sparse.csc_matrix((profiles, indexes), shape=(npts, n_features))
    return mat



class SaturatedVoigtFeatureModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    
    def __init__(self, 
                 transitions, 
                 min_wv,
                 max_wv,
                 snr_threshold=5.0,
                 snr_target=100.0,
                 dof_thresholds=None,
                 teff=5000.0,
                 gamma=0.01, 
                 vmicro=2.0,
                 initial_x_offset=7.0,
                 dof_threshold=0.0,
                 log_ion_frac=-0.01,
                 max_delta_wv=100,
                 max_delta_x=0.25,
                 domination_ratio=5.0,
                 H_mask_radius=2.0,
                 species_grouper="unique",
                 model_resolution=2e5,
                 ):
        """provides a rough model of a normalized stellar absorption spectrum
        
        inputs 
        
        transitions: pandas.DataFrame
          a DataFrame containing all all the transition data
        min_wv: float
          mininum model wavelength
        max_wv: float
          maximum model wavelength
        teff: float
          effective temperature in Kelvin
        vmicro: float
          microturbulent velocity in kilometers per second
        vmacro: float
          macroturbulent velocity in kilometers per second
        gamma: float
          the ratio of the lorentz width to the thermal width at 5000 angstroms.
        """
        super(SaturatedVoigtFeatureModel, self).__init__()
        self.fdat = transitions._data.copy()
        self.n_feat = len(self.fdat)
        self.min_wv = float(min_wv)
        self.max_wv = float(max_wv)
        self._teff = teff
        self.snr_threshold = snr_threshold
        self.snr_target = snr_target
        self._vmicro = vmicro
        assert log_ion_frac < 0 #can't have an ion fraction greater than 1!
        self.log_ion_frac = log_ion_frac
        self.domination_ratio=domination_ratio
        self.H_mask_radius=H_mask_radius
        self.max_delta_wv = max_delta_wv
        self.max_delta_x = max_delta_x
        
        self._check_initialize_column("ew", fill_value=0.001)
        self._check_initialize_column("wv_offset", fill_value=0.0)
        self._check_initialize_column("sigma_offset", fill_value=0.0)
        self._check_initialize_column("gamma_offset", fill_value=0.0)
        self._check_initialize_column("x_offset", fill_value=initial_x_offset)
        self._check_initialize_column("group_dof", 0.0)
        self._check_initialize_column("group_exemplar", 0)
        self._check_initialize_column("feature_num", np.arange(self.n_feat))
        self._check_initialize_column("fit_group", 0)
        self._check_initialize_column("species_group", 0)
        
        if isinstance(species_grouper, basestring):
            species_grouper = SpeciesGrouper([], ungrouped_val=species_grouper) 
        self.species_grouper = species_grouper
        
        #TODO: make the gamma ratio change as a function of wv
        #TODO: allow for individual gamma ratio's by interpolating cogs
        self.gamma = gamma
        self.model_resolution=model_resolution
        
        self.generate_fit_groups()
        self.calc_all()
        self.npts = int(np.log(self.max_wv/self.min_wv)*self.model_resolution)
        self.model_wv = np.exp(np.linspace(np.log(self.min_wv), np.log(self.max_wv), self.npts))
        self.model_wv_gradient = scipy.gradient(self.model_wv)
        self.delta_log = np.log(self.max_wv/self.min_wv)/self.npts
        #a is defined as gamma*wv**2/(4*pi*c*doppler_width)
        
        self.calc_feature_matrix(overwrite=True)
        self.calc_relative_opac_matrix(overwrite=True)
        self.collapse_feature_matrix(overwrite=True)
    
    def _check_initialize_column(self, col_name, fill_value):
        """check if a column is defined and if it is not add it"""
        if not (col_name in self.fdat.columns):
            if len(np.atleast_1d(fill_value)) > 1:
                new_col = fill_value    
            else:
                new_col = np.empty(self.n_feat, dtype=float)
                new_col.fill(fill_value)
            self.fdat[col_name] = new_col
    
    def calc_all(self):
        self.calc_cog()
        self.calc_solar_ab()
        self.calc_doppler_widths()
        self.calc_x()
    
    def __call__(self, input, **kwargs):
        if self._recalc_doppler_widths:
            self.calc_doppler_widths()
        if self._recalc_x:
            self.calc_x()
        if self._recalc_opac_matrix:
            self.calc_relative_opac_matrix()
        if self._recalc_feature_matrix:
            self.calc_feature_matrix()
        if self._recollapse_feature_matrix:
            self.collapse_feature_matrix()
        ret_val = self.cfm*self.exemplar_opac_p.get()
        return ret_val
    
    #@parameter(free=True, scale=1.0, min=0.0001, max=1000.0, step_scale=1.0)
    def exemplar_opac_p(self, ):
        return self.exemplar_opac
    
    #@exemplar_opac_p.setter
    def set_exemplar_opac(self, exemplar_opacities):            
        self.exemplar_opac = exemplar_opacities
        per_feature_opacities = self.opac_matrix*exemplar_opacities
        self.fdat["opac_strength"] = per_feature_opacities
    
    #@exemplar_opac_p.expander
    def expand_exemplar_effects(self, input_vec, **kwargs):
        return self.cfm
    
    #@parameter(free=True, min=0.001, max=1.2, step_scale=0.1)
    def gamma_p(self):
        return self.gamma
    
    #@gamma_p.setter
    def set_gamma(self, value):
        #import pdb; pdb.set_trace()
        self.gamma = value
        self._recalc_opac_matrix = True
        self._recalc_feature_matrix = True
        self._recollapse_feature_matrix = True
    
    #def calc_ew_errors(self):
    #    self.fdat["ew_error"] = np.ones(len(self.fdat))
    #    total_sig2 = self.fdat["ew"]
    #    ew_error = np.sqrt(self.fdat["ew_frac"]*self.fdat["ew"])
    #    self.fdat["ew_error"] = np.where(self.fdat["ew_frac"])
    
    def group_species(self, overwrite=True):
        species_nums = self.species_grouper(self.fdat.species.values)
        if overwrite:
            self.fdat["species_group"] = species_nums
            unique = np.unique(self.fdat.species_group.values)
            self.species_id_set = np.sort(unique)
        return species_nums
    
    def generate_fit_groups(self):
        #import pdb; pdb.set_trace()
        self.group_species(overwrite=True)
        species_gb = self.fdat.groupby("species_group")
        species_groups = species_gb.groups
        n_total_groups = 0
        for species_key in species_groups:
            species_ixs = np.asarray(species_groups[species_key])
            n_lines = len(species_ixs)
            new_fit_groups = []
            targ_n_group = 5
            if n_lines < 2*targ_n_group:
                new_fit_groups.append((species_ixs, 0))
            else:
                species_dat = self.fdat.ix[species_ixs]
                cspecies_pnum = species_dat["Z"].iloc[0]
                cabund = ptable["abundance"].ix[(cspecies_pnum, 0)]
                rel_strengths = cabund + species_dat["loggf"] - self.theta*species_dat["ep"]
                strength_sorted_ixs = species_ixs[np.argsort(rel_strengths).values]
                sorted_strengths = rel_strengths[strength_sorted_ixs]
                cumulative_strength = np.zeros(len(sorted_strengths)+1)
                cumulative_strength[1:] = np.cumsum(10**sorted_strengths)
                targ_cum_strength = 18.0
                sub_lb, sub_ub = 0, 0
                grp_id = 0
                accum_str = 0
                while sub_ub < n_lines:
                    while (accum_str < targ_cum_strength) and (sub_ub < n_lines):
                        sub_delta = sub_ub-sub_lb
                        if sub_delta < targ_n_group:
                            sub_ub = min(sub_ub + targ_n_group, n_lines)
                        else:
                            sub_ub += 1
                        accum_str = cumulative_strength[sub_ub]-cumulative_strength[sub_lb]
                    if (n_lines - sub_ub) <= targ_n_group:
                        #we wont' be able to achieve the target number of lines just make this the end
                        sub_ub = n_lines
                    new_fit_groups.append((strength_sorted_ixs[sub_lb:sub_ub], grp_id))
                    grp_id += 1
                    accum_str = 0
                    sub_lb = sub_ub
            #now process all the groups and generate a unique exemplar for each
            for fg, fg_idx in new_fit_groups:
                n_total_groups += 1
                self.fdat.ix[fg, "fit_group"] = fg_idx 
                exemplar_idx = np.random.randint(len(fg))
                exemplar_vec = np.zeros(len(fg), dtype=int)
                exemplar_vec[exemplar_idx] = 1
                self.fdat.ix[fg, "group_exemplar"] = exemplar_vec 
        
        self.exemplar_opac = np.repeat(0.01, n_total_groups)
    
    def optimize_fit_groups(self, spectra):
        transforms_to_model = []
        npts_model = len(self.model_wv)
        min_delta_wv = np.empty(npts_model, dtype=float)
        min_delta_wv.fill(100.0) #bigger than any reasonable pixel size
        snr2_available = np.zeros(npts_model)
        dof_available = np.zeros(npts_model)
        
        logger("building spectra to model space transforms")
        for spec in spectra:
            trans = tmb.utils.resampling.get_resampling_matrix(spec.wv, self.model_wv, preserve_normalization=False)
            transforms_to_model.append(trans)
            cur_snr2 = trans*(spec.flux**2*spec.ivar)
            cur_dof = trans*np.ones(spec.wv.shape)
            max_dof = np.max(cur_dof)
            cur_dof /= max_dof #TODO replace this with a preserved norm
            cur_delta_pix = trans*scipy.gradient(spec.wv)/max_dof 
            cur_delta_pix = np.where(cur_delta_pix > 0, cur_delta_pix, 100.0)
            min_delta_wv = np.where(cur_delta_pix < min_delta_wv, cur_delta_pix, min_delta_wv)
            snr2_available += cur_snr2
            dof_available = np.where(dof_available > cur_dof, dof_available, cur_dof)
        
        #get the log of the min pixel size as a fraction of wavelength
        pixel_lrw = np.log10(min_delta_wv/self.model_wv)
        
        #admit features up to 1/100 of a pixel width in predicted equivalent width
        max_strengths = np.power(10.0, pixel_lrw - 2.0) 
        
        #zero is the background group
        self.fdat["fit_group"] = np.zeros(len(self.fdat), dtype=int) 
        window_delta = 0.0001
        logger("collecting max feature strengths per model pixel")
        for line_idx in range(len(self.fdat)):
            ldata = self.fdat.iloc[line_idx]
            cent_wv = ldata["wv"]
            if cent_wv > self.max_wv:
                continue
            if cent_wv < self.min_wv:
                continue
            min_idx = self.get_index(cent_wv*(1.0-window_delta), clip=True)
            max_idx = self.get_index(cent_wv*(1.0+window_delta), clip=True)
            sigma = ldata["doppler_width"]
            #TODO: replace the constant gamma with an appropriately varying one
            gamma = self.gamma*sigma
            prof = voigt(self.model_wv[min_idx:max_idx+1], cent_wv, sigma, gamma)
            predicted_lrw = ldata["cog_lrw"]
            prof *= np.power(10.0, predicted_lrw)/np.max(prof)
            mst_snippet = max_strengths[min_idx:max_idx+1]
            max_strengths[min_idx:max_idx+1] = np.where(mst_snippet > prof, mst_snippet, prof)
        
        logger("relegating dominated features to a background model")
        #mask features close to the centers of Hydrogen lines
        hmask = hydrogen.get_H_mask(self.model_wv, self.H_mask_radius)
        for line_idx in range(len(self.fdat)):
            ldata = self.fdat.iloc[line_idx]
            cent_wv = ldata["wv"]
            if cent_wv > self.max_wv:
                continue
            if cent_wv < self.min_wv:
                continue
            cent_idx = self.get_index(cent_wv, )
            if not hmask[cent_idx]: 
                continue #too close to a hydrogen feature
            predicted_lrw = ldata["cog_lrw"]
            if max_strengths[cent_idx] < self.domination_ratio*np.power(10.0, predicted_lrw):
                self.fdat["fit_group"].iloc[line_idx] = 1
        
        #print "beginning grouping"
        logger("matching potential groups")
        foreground_features = self.fdat[self.fdat.fit_group >= 1]
        
        species_gb = foreground_features.groupby(["Z", "ion"])
        all_fit_groups = []
        species_groups = species_gb.groups
        for species_pair in species_gb.groups:
            species_ixs = species_groups[species_pair]
            species_df = foreground_features.ix[species_ixs]
            wv_x_vals = species_df[["wv", "x"]].values.copy()
            wv_x_vals[:, 0] /= self.max_delta_wv
            wv_x_vals[:, 1] /= self.max_delta_x
            m1, m2, dist = matching.match(wv_x_vals, wv_x_vals, tolerance=1.0) 
            potential_groups = [[]]
            last_matching_idx = m1[0]
            for match_idx in range(len(m1)):
                if last_matching_idx != m1[match_idx]:
                    potential_groups.append([])
                    last_matching_idx = m1[match_idx]
                potential_groups[-1].append(m2[match_idx])
            
            used_features = set()
            #for each feature attempt to build a suitable group
            for potential_group in potential_groups:
                potential_ixs = [species_ixs[pgi] for pgi in potential_group]                
                group_ixs = []
                accum_snr2 = 0.0
                accum_dof = 0.0
                to_subtract = []
                for fdat_ix in potential_ixs:
                    if fdat_ix in used_features:
                        continue
                    group_ixs.append(fdat_ix)
                    ldata = self.fdat.ix[fdat_ix]
                    cent_wv = ldata["wv"]
                    cent_idx = self.get_index(cent_wv)
                    min_idx = self.get_index(cent_wv*(1.0-window_delta), clip=True)
                    max_idx = self.get_index(cent_wv*(1.0+window_delta), clip=True)
                    sigma = max(ldata["doppler_width"], min_delta_wv[cent_idx])
                    #TODO: replace the constant gamma with an appropriately varying one
                    gamma = self.gamma*sigma
                    cog_lrw = ldata["cog_lrw"]
                    cog_ew = np.power(10.0, cog_lrw)*cent_wv
                    prof = cog_ew*voigt(self.model_wv[min_idx:max_idx+1], cent_wv, sigma, gamma)
                    prof = np.where(prof < 1, prof, 1.0) #make sure we don't run to negative flux
                    snr2_prod = snr2_available[min_idx:max_idx+1]*prof
                    captured_snr2 = np.sum(snr2_prod)
                    accum_snr2 += captured_snr2
                    nprof = prof/np.max(prof)
                    dof_prod = dof_available[min_idx:max_idx+1]*nprof
                    accum_dof += np.sum(dof_prod)
                    to_subtract.append((min_idx, max_idx, dof_prod, snr2_prod))
                    add_group = False
                    meets_snr_threshold = accum_snr2 > self.snr_threshold**2
                    meets_snr_target = accum_snr2 > self.snr_target**2
                    non_degen = accum_dof > 1.0
                    is_last_match = fdat_ix == potential_ixs[-1]
                    if meets_snr_target and non_degen:
                        add_group = True
                    elif meets_snr_threshold and non_degen and is_last_match:
                        add_group = True
                    if add_group:
                        print "accum n {}, accum dof {: 5.2f} accum snr {: 5.2f}".format(len(group_ixs), accum_dof, np.sqrt(accum_snr2))
                        #this is a good enough group add it to the set of groups
                        all_fit_groups.append(group_ixs)
                        #pick a group exemplar at random
                        exemplar_ix = group_ixs[np.random.randint(0, len(group_ixs))]
                        #set their fit group column to match each other
                        for feat_ix in group_ixs:
                            if feat_ix == exemplar_ix:
                                self.fdat.ix[feat_ix, "group_exemplar"] = 1
                            else:
                                self.fdat.ix[feat_ix, "group_exemplar"] = 0
                            self.fdat.ix[feat_ix, "fit_group"] = len(all_fit_groups)+1
                            used_features.add(feat_ix)
                        #subtract from our total allotment of signal to noise and degrees of freedom
                        for min_idx, max_idx, dof_prod, snr2_prod in to_subtract:
                            resid_snr2 = snr2_available[min_idx:max_idx+1]-snr2_prod
                            snr2_available[min_idx:max_idx+1] = np.where(resid_snr2 > 0, resid_snr2, 0.0)
                            resid_dof = dof_available[min_idx:max_idx+1] - dof_prod*min(1.0, 3.0/len(to_subtract))
                            dof_available[min_idx:max_idx+1] = np.where(resid_dof > 0, resid_dof, 0.0)
                    if is_last_match:
                        if not add_group:
                            self.fdat.ix[potential_ixs[0], "fit_group"] = 0
        
    def get_index(self, wv, clip=False):
        idx = np.log(np.asarray(wv)/self.min_wv)/self.delta_log
        if clip:
            idx = np.clip(idx, 0, self.npts-1)
        return idx
    
    def sigma_widths(self):
        sig_offs= self.fdat["sigma_offset"]
        return np.clip(self.fdat["doppler_width"]+sig_offs, 0.0001, np.inf)
    
    def gamma_widths(self):
        gam_offs = self.fdat["gamma_offset"]
        return np.clip(self.gamma*(self.fdat["wv"]/5000.0)**2+gam_offs, 0.0001, np.inf)
    
    def calc_feature_matrix(self, overwrite=True):
        print "generating full feature matrix"
        wvs = self.fdat["wv"].values
        sigmas = self.sigma_widths().values
        gammas = self.gamma_widths().values
        full_matrix = voigt_feature_matrix(self.model_wv, wvs, sigmas, gammas)
        if overwrite:
            self.feature_matrix = full_matrix
            self._recalc_feature_matrix = False 
        return full_matrix 
    
    def calc_relative_opac_matrix(self, overwrite=True):
        grouping = ["species_group", "fit_group"]
        group_gb = self.fdat.groupby(grouping)
        groups = group_gb.groups
        n_groups = len(groups)
        group_keys = sorted(groups.keys())
        opac_frac = pd.Series(np.zeros(len(self.fdat)), index=np.array(self.fdat.index))
        matrix_idxs = {species:[] for species in self.species_id_set}
        opac_frac_data = {species:[] for species in self.species_id_set}
        col_max = {}
        for group_idx in range(n_groups):
            group_id = group_keys[group_idx]
            #TODO: remove background features by detecting negative fit groups
            group_ldf = self.fdat.ix[groups[group_id]]
            rel_opacs = np.exp(group_ldf["x"])
            opac_sum = np.sum(rel_opacs)
            rel_opacs /= opac_sum
            opac_frac.ix[groups[group_id]] = rel_opacs
            group_feat_nums = self.fdat.ix[groups[group_id], "feature_num"].values
            n_group_feats = len(group_feat_nums)
            grp_col = int(group_id[1])
            col_max[group_id[0]] = grp_col+1 #this will end up the max since we sort the keys
            midxs = np.array([group_feat_nums, np.repeat(grp_col, n_group_feats)])
            matrix_idxs[group_id[0]].append(midxs)
            opac_frac_data[group_id[0]].append(rel_opacs.values)
        matrix_idxs = {species:np.hstack(matrix_idxs[species]) for species in self.species_id_set}
        ew_frac_data = {species:np.hstack(opac_frac_data[species]) for species in self.species_id_set}
        grouping_matrices = []
        for species in self.species_id_set:
            n_sub_groups = col_max[species]
            n_grouped_feats = matrix_idxs[species].shape[1]
            mdat = ew_frac_data[species]
            idx_dat = matrix_idxs[species]
            gmat = scipy.sparse.csc_matrix((mdat, idx_dat), shape=(self.n_feat, n_sub_groups))
            grouping_matrices.append(gmat)
        grouping_matrix = scipy.sparse.bmat([grouping_matrices]) 
        if overwrite:
            self.opac_matrix = grouping_matrix 
            self._recalc_opac_matrix=False
            self._recollapse_feature_matrix=True
        return grouping_matrix
    
    def collapse_feature_matrix(self, overwrite=True):
        #print "collapsing to group feature matrices"
        cfm = self.feature_matrix*self.opac_matrix
        if overwrite:
            self.cfm = cfm
            self._recollapse_feature_matrix = False
        return cfm
    
    @property
    def teff(self):
        return self._teff
    
    @teff.setter
    def teff(self, value):
        self._teff = value
        self._recalc_doppler_widths = True
        self._recalc_x = True
        self._recalc_cog_ews = True
        self._recalc_opac_matrix = True
        self._recalc_feature_matrix=True
        self._recollapse_feature_matrix = True
    
    @property
    def vmicro(self):
        return self._vmicro
    
    @vmicro.setter
    def vmicro(self, value):
        self._vmicro = value
        self._recalc_doppler_widths = True
        self._recalc_x = True
        self._recalc_cog_ews = True
        self._recalc_opac_matrix = True
        self._recalc_feature_matrix=True
        self._recollapse_feature_matrix = True
    
    #@parameter(free=True, min=0.01, max=10.0, step_scale=0.1)
    def vmicro_p(self):
        return self.vmicro
    
    #@vmicro_p.setter
    def set_vmicro(self, value):
        self.vmicro = value
    
    @property
    def theta(self):
        return 5040.0/self.teff
    
    @theta.setter
    def theta(self, value):
        self.teff = 5040.0/value
    
    #@parameter(free=False, min=0.2, max=3.0, step_scale=0.1)
    def theta_p(self):
        return self.theta
    
    #@theta_p.setter
    def set_theta(self, value):
        self.theta = value
    
    def calc_solar_ab(self):
        self.fdat["solar_ab"] = ptable["abundance"].ix[(self.fdat.z, 0)]
    
    def calc_therm_widths(self):
        #TODO: use real atomic weight
        weight =2.0*self.fdat["Z"]
        wv = self.fdat["wv"]
        widths = 4.301e-7*np.sqrt(self.teff/weight)*wv
        self.fdat["thermal_width"] = widths
    
    def calc_vmicro_widths(self):
        self.fdat["vmicro_width"] = self.fdat["wv"]*self.vmicro/299792.458
    
    def calc_doppler_widths(self):
        #"print calculating doppler widths"
        self.calc_therm_widths()
        self.calc_vmicro_widths()
        dop_widths = np.sqrt(self.fdat["vmicro_width"]**2 + self.fdat["thermal_width"]**2)
        self.fdat["doppler_width"] = dop_widths
        self._recalc_doppler_widths = False
    
    def calc_x(self):
        #"print calculating x values"
        neutral_delt_x = np.log10(1.0 - np.power(10.0, self.log_ion_frac))
        ionization_delta = np.where(self.fdat.ion == 1, self.log_ion_frac, neutral_delt_x)
        #TODO: include some sort of basic adjustment of ionization fraction on the basis of element ionization energy.
        self.fdat["x"] = self.fdat.solar_ab + self.fdat.loggf - self.fdat.ep*self.theta + ionization_delta
        self._recalc_x = False
        
    def calc_cog(self):
        """calculate approximate curves of growth to convert from opacity to EW
        """
        self._log_cog_gamma_ratios = np.linspace(-5.0, 0.5, 31)
        self._cogs = [saturated_voigt_cog(gamma_ratio=10**grat) for grat in self._log_cog_gamma_ratios]
        self._min_log_cog = self._log_cog_gamma_ratios[0]
        self._delta_log_cog = self._log_cog_gamma_ratios[1]-self._min_log_cog
    
    def calc_ew(self):
        sigma_widths = self.sigma_widths()
        gamma_widths = self.gamma_widths() 
        log_gamma_ratios = np.log10(gamma_widths/sigma_widths)
        cog_idxs = np.array((log_gamma_ratios - self._min_log_cog)/self._delta_log_cog, dtype=int)
        cog_idxs = np.clip(cog_idxs, 0, len(self._cogs)-1)
        log_rews_out = np.zeros(len(self.fdat))
        opac_strength = self.fdat["opac_strength"].values
        for cur_cog_idx in range(len(self._cogs)):
            select_idxs = np.where(cog_idxs == cur_cog_idx)[0]
            if len(select_idxs) > 0:
                log_rews_out[select_idxs] = self._cogs[cur_cog_idx](np.log10(opac_strength[select_idxs]))
        ews = sigma_widths*np.power(10.0, log_rews_out)
        self.fdat["ew"] = ews
    
    #def fit_offsets(self):
    #    #import pdb; pdb.set_trace()
    #    species_gb = self.fdat.groupby("species_group")
    #    groups = species_gb.groups
    #    #order the species keys so we do the species with the most exemplars first
    #    #num_exemplars = sorted([(len(groups[k]), k) for k in groups.keys()], reverse=True)
    #    fallback_offset = 10.0 #totally arbitrary
    #    for species_key in range(len(groups)):
    #        species_df = self.fdat.ix[groups[species_key]]
    #        exemplars = species_df[species_df.group_exemplar > 0]
    #        delta_rews = np.log10(exemplars.ew/exemplars.doppler_width)
    #        x_deltas = exemplars.x.values - self.cog.inverse(delta_rews.values)
    #        offset = np.sum(x_deltas)
    #        if np.isnan(offset):
    #            offset = fallback_offset
    #        self.fdat.ix[groups[species_key], "x_offset"] = offset
    
    @property
    def doppler_lrw(self):
        return np.log10(self.fdat.doppler_width/self.fdat.wv)
    
    @property
    def lrw(self):
        return np.log10(self.fdat["ew"]/self.fdat["wv"])
    
    #@property
    #def x_adj(self):
    #    return self.fdat.x - self.fdat.x_offset
    #
    #@property
    #def sigma(self):
    #    sig = np.sqrt(self.fdat["vmicro_width"]**2 
    #            + self.fdat["thermal_width"]**2 
    #            + (self.fdat["wv"]*self.vmacro/299792.458)**2) 
    #    return sig + self.fdat["sigma_off"]
    
    @property
    def lrw_adj(self):
        return self.lrw - self.doppler_lrw
    
    def cog_plot(self, ax=None, **kwargs):
        foreground = self.fdat[(self.fdat.fit_group > 1)*(self.fdat.group_exemplar > 0)]
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(foreground.x-foreground.x_offset, np.log10(foreground.ew/foreground.doppler_width), **kwargs)
        ax.set_xlabel("adjusted relative strength")
        ax.set_ylabel("log(EW/doppler_width)")

class SimpleMatrixOpacityModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    
    def __init__(self, model_wvs, opac_matrix, opac_strength):
        Model.__init__(self)
        self.wv = model_wvs
        self.opac_matrix = scipy.sparse.csr_matrix(opac_matrix)
        assert len(opac_matrix)==len(model_wvs)
        self.opac_strength = opac_strength
    
    #@parameter(free=True, min=0.0001)
    def opac_strength_p(self):
        return self.opac_strength
    
    #@opac_strength_p.setter
    def opac_strength(self, value):
        self.opac_strength = value
    
    def __call__(self, input_vec):
        return self.opac_matrix*self.opac_strength + input_vec

class OpacityToTransmission(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    
    def __init__(self):
        Model.__init__(self)
    
    def __call__(self, input_vec):
        return np.exp(-input_vec)
    
    def as_linear_op(self, input_vec):
        npts = len(input_vec)
        return scipy.sparse.dia_matrix((-np.exp(-input_vec), 0), shape=(npts, npts))

class Feature(object):
    
    def __init__(self, 
                 profile, 
                 eq_width, 
                 abundance, 
                 trans_parameters,
                 relative_continuum=1.0, 
                 data_sample=None,
                 flags=None,
                 note=""):
        self.profile = profile
        self._eq_width = eq_width
        self.abundance = abundance
        self.trans_parameters = trans_parameters
        self.data_sample = data_sample
        if flags == None:
            flags = FeatureFlags()
        self.flags = flags
        self.note=note
        self.relative_continuum=relative_continuum
    
    def __repr__ (self):
        rep_str = """Feature : %s notes: %s"""
        return rep_str % (repr(self.trans_parameters), self.notes) 
    
    def chi_sq_vec(self, wvs=None):
        if wvs == None:
            wvs = self.data_sample.wv
    
    def trough_bounds(self, wvs, fraction=0.95):
        pass
    
    def get_offset(self):
        return self.profile.get_parameters()[0]
    
    def set_offset(self, new_off):
        cur_p = self.profile.get_parameters()
        cur_p[0] = new_off
        self.profile.set_parameters(cur_p)
    
    @property
    def molecular_weight(self):
        return self.trans_parameters.molecular_weight
    
    @property
    def wv(self):
        return self.trans_parameters.wv
    
    @property
    def species(self):
        return self.trans_parameters.species
    
    @property
    def loggf(self):
        return self.trans_parameters.loggf
    
    @property
    def ep(self):
        return self.trans_parameters.ep
    
    def get_model_flux(self, wvs):
        return self.model_flux(wvs)
    
    def model_flux(self, wvs):
        return self.relative_continuum*(1.0-self.eq_width*self.profile(wvs))
    
    @property
    def eq_width(self):
        return self._eq_width
    
    @property
    def depth(self):
        return self.eq_width_to_depth(self._eq_width)
    
    def eq_width_to_depth(self, eq_width):
        coff = self.profile.get_parameters()[0]
        cdepth = self.profile(self.wv+coff)
        return eq_width*cdepth
    
    def depth_to_eq_width(self, depth):
        coff = self.profile.get_parameters()[0]
        cdepth = self.profile(self.wv+coff)
        return depth/cdepth
    
    def set_depth(self, depth):
        eqw = self.depth_to_eq_width(depth)
        self.set_eq_width(eqw)
    
    def set_eq_width (self,eq_width):
        self._eq_width = eq_width
    
    def set_relative_continuum(self, rel_cont):
        self.relative_continuum = rel_cont
    
    @property
    def logrw(self):
        return np.log10(self.eq_width/self.wv)
    
    def thermal_width(self, teff):
        4.301e-7*np.sqrt(teff/self.molecular_weight)*self.wv
    
    def get_cog_point(self, teff, vturb=2.0, abundance_offset=0.0):
        #most likely velocity
        vml = np.sqrt(self.thermal_width(teff)**2 + vturb**2) 
        solar_logeps = tmb.stellar_atmospheres.solar_abundance[self.species]["abundance"]
        theta = 5040.0/teff
        x = solar_logeps+self.loggf-self.ep*theta
        x += abundance_offset
        logvml = np.log10(vml)
        x -= logvml
        y = self.logrw - logvml
        return x, y


class AtomicTransition:
    """
    Holds parameters for a specific energy transition
    
    wavelength,id_,loggf,ep,vwdamp=0,d0=0,info=None
    Parameters
    ----------
    wavelength : float
        Gives the wavelength, in Angstroms, of the transition 
    id_ : float
        Gives the transition id, for atomic transitions this is the species 
        # TODO: have solar abundance convert string to integer for this
    loggf : float
        Oscillator strength of the transition
    ep : float
        The excitation potential of the transition
    vwdamp : float
        The VanDer Waals Damping constant for the transition
    d0 : float
        The dissociation energy for the transition
    
    Raises
    ------
    none
    
    
    Notes
    -----
    __1)__ none
    
    
    Examples
    --------
    >>> transition = TransitionProperties(5555.5,26.0,4.0,-1.34)
    >>>
    >>>
    
    """
    
    def __init__ (self,wavelength,id_,loggf,ep,vwdamp=0,d0=0):
        self.wv = wavelength
        # if id_ is given as string (e.g. 'Fe I') then this will get the 
        # appropriate id
        #if isinstance(id_,basestring):
        #    id_ = periodic_table[id_][0] 
        self._id = id_
        self.loggf = loggf
        self.ep = ep
        self.vwdamp = vwdamp
        self.d0 = d0
    
    @property
    def molecular_weight(self):
        """molecular weight in amu of the species"""
        #TODO: use the real molecular weight instead of the species number
        return np.around(2*self._id)
    
    @property
    def species(self):
        return self._id
    
    def __repr__ (self):
        out = (format(self.wv,'10.3f'),
               format(self._id,'5.1'),
               format(self.loggf,'5.2f'),
               format(self.ep,'5.2f'))
        return "  ".join(out)
    
