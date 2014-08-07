import numpy as np
import pandas as pd
import thimbles as tmb
import matplotlib.pyplot as plt
import scipy
import scipy.integrate as integrate
import scipy.sparse
from flags import FeatureFlags
from thimbles.stellar_atmospheres import solar_abundance as ptable
from thimbles.profiles import voigt
from thimbles.utils.misc import smooth_ppol_fit
import thimbles.utils.piecewise_polynomial as ppol
from thimbles import verbosity
from thimbles import hydrogen
from latbin import matching

class SaturatedVoigtFeatureModel(object):
    
    def __init__(self, 
                 transitions, 
                 min_wv,
                 max_wv,
                 snr_threshold=1.0,
                 teff=5000.0, 
                 vmicro=2.0, 
                 vmacro=1.0, 
                 gamma_ratio_5000=0.02,
                 initial_x_offset=8.0,
                 delta_wv=50,
                 delta_x=0.25,
                 domination_ratio=4.0,
                 param_hierarchy=None,
                 H_mask_radius=2.0,
                 model_resolution=5e5,):
        """provides a model of a normalized absorption spectrum
        
        transitions: pandas.DataFrame
          a DataFrame containing all all the transition data
        teff: float
          effective temperature in Kelvin
        vmicro: float
          microturbulent velocity in kilometers per second
        vmacro: float
          macroturbulent velocity in kilometers per second
        gamma_ratio_5000: float
          the ratio of the lorentz width to the thermal width at 5000 angstroms.
        """
        if param_hierarchy is None:
            param_hierarchy = {}
            #param_hierarchy["param_name"] = accumulated_group_degrees_of_freedom_threshold
            param_hierarchy["ew"] = 0.5
            param_hierarchy["vel_offset"] = 2.0
            param_hierarchy["sigma_offset"] = 3.0
            param_hierarchy["gamma_offset"] = 5.0
        self.param_hierarchy = param_hierarchy
        self.min_wv = float(min_wv)
        self.max_wv = float(max_wv)
        self.teff = teff
        self._initial_x_off = initial_x_offset
        self.snr_threshold = snr_threshold
        self.vmicro = vmicro
        self.vmacro = vmacro
        self.domination_ratio=domination_ratio
        self.H_mask_radius=H_mask_radius
        self.delta_wv = delta_wv
        self.delta_x = delta_x
        #TODO: make the gamma ratio change as a function of wv
        #TODO: allow for individual gamma ratio's by interpolating cogs
        self.gamma_ratio_5000 = gamma_ratio_5000
        self.model_resolution=model_resolution
        
        self.npts = int(np.log(self.max_wv/self.min_wv)*self.model_resolution)
        self.model_wv = np.exp(np.linspace(np.log(self.min_wv), np.log(self.max_wv), self.npts))
        self.model_wv_gradient = scipy.gradient(self.model_wv)
        self.delta_log = np.log(self.max_wv/self.min_wv)/self.npts
        
        self.fdat = transitions.copy()
        self.n_feat = len(self.fdat)
        self._initialize_columns()
        #a is defined as gamma*wv**2/(4*pi*c*doppler_width)
    
    def _initialize_columns(self):
        cur_columns = self.fdat.columns 
        targ_columns = ["ew", "wv_offset", "sigma_offset", "gamma_offset"]
        for col in targ_columns:
            if not (col in cur_columns):
                self.fdat[col] = np.zeros(self.n_feat)
        
        self.fdat["gamma_ratio"] = np.ones(self.n_feat)*self.gamma_ratio_5000
        self.fdat["x_offset"] = np.ones(self.n_feat)*self._initial_x_off
        self.calc_cog()
        self.calc_solar_ab()
        self.calc_doppler_widths()
        self.calc_x()
        self.calc_cog_lrws()
    
    def __getattr__(self, attr_name):
        return eval("self.fdat['{}']".format(attr_name))
    
    def optimize_fit_groups(self, spectra):
        transforms_to_model = []
        npts_model = len(self.model_wv)
        min_delta_wv = np.empty(npts_model, dtype=float)
        min_delta_wv.fill(100.0) #bigger than any reasonable pixel size
        snr2_available = np.zeros(npts_model)
        dof_available = np.zeros(npts_model)
        
        verbosity("building spectra to model space transforms")
        for spec in spectra:
            trans = tmb.utils.resampling.get_resampling_matrix(spec.wv, self.model_wv, preserve_normalization=False)
            transforms_to_model.append(trans)
            cur_snr2 = trans*(spec.flux**2*spec.inv_var)
            cur_dof = trans*np.ones(spec.wv.shape)
            max_dof = np.max(cur_dof)
            cur_dof /= max_dof #TODO replace this with a preserved norm
            cur_delta_pix = trans*scipy.gradient(spec.wv)/max_dof 
            cur_delta_pix = np.where(cur_delta_pix > 0, cur_delta_pix, 100.0)
            min_delta_wv = np.where(cur_delta_pix < min_delta_wv, cur_delta_pix, min_delta_wv)
            snr2_available += cur_snr2
            dof_available = np.where(dof_available > cur_dof, dof_available, cur_dof)
        
        #import pdb; pdb.set_trace()
        #get the log of the min pixel size as a fraction of wavelength
        pixel_lrw = np.log10(min_delta_wv/self.model_wv)
        
        #admit features up to 1/100 of a pixel width in predicted equivalent width
        max_strengths = np.power(10.0, pixel_lrw - 2.0) 
        
        #zero is the background group
        self.fdat["fit_group"] = np.zeros(len(self.fdat), dtype=int) 
        window_delta = 0.0001
        verbosity("collecting max feature strengths per model pixel")
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
            gamma = self.gamma_ratio_5000*sigma
            prof = voigt(self.model_wv[min_idx:max_idx+1], cent_wv, sigma, gamma)
            predicted_lrw = ldata["cog_lrw"]
            prof *= np.power(10.0, predicted_lrw)/np.max(prof)
            mst_snippet = max_strengths[min_idx:max_idx+1]
            max_strengths[min_idx:max_idx+1] = np.where(mst_snippet > prof, mst_snippet, prof)
        
        verbosity("relegating dominated features to a background model")
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
        verbosity("matching potential groups")
        foreground_features = self.fdat[self.fdat.fit_group >= 1]
        
        species_gb = foreground_features.groupby(["Z", "ion"])
        all_fit_groups = []
        species_groups = species_gb.groups
        for species_pair in species_gb.groups:
            species_ixs = species_groups[species_pair]
            species_df = foreground_features.ix[species_ixs]
            wv_x_vals = species_df[["wv", "x"]].values.copy()
            wv_x_vals[:, 0] /= self.delta_wv
            wv_x_vals[:, 1] /= self.delta_x
            m1, m2, dist = matching.match(wv_x_vals, wv_x_vals, tolerance=1.0) 
            match_idx = 0
            feature_idx = 0
            used_features = set()
            #for each feature attempt to build a suitable group
            while feature_idx < len(species_df):
                #ignore features already in a group
                if feature_idx in used_features:
                    feature_idx += 1
                    continue
                group_ixs = []
                accum_snr2 = 0.0
                accum_dof = 0.0
                to_subtract = []
                #put every allowed pair for this feature into a group with it and see if it is good enough
                while (match_idx < len(m1)) and (m1[match_idx] == feature_idx):
                    fdat_ix = species_ixs[m2[match_idx]]
                    if fdat_ix in used_features:
                        match_idx +=1
                        continue
                    group_ixs.append(fdat_ix)
                    ldata = self.fdat.ix[fdat_ix]
                    cent_wv = ldata["wv"]
                    cent_idx = self.get_index(cent_wv)
                    min_idx = self.get_index(cent_wv*(1.0-window_delta), clip=True)
                    max_idx = self.get_index(cent_wv*(1.0+window_delta), clip=True)
                    sigma = max(ldata["doppler_width"], min_delta_wv[cent_idx])
                    #TODO: replace the constant gamma with an appropriately varying one
                    gamma = self.gamma_ratio_5000*sigma
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
                    to_subtract.append((min_idx, max_idx, snr2_prod, dof_prod))
                    match_idx += 1
                #print "accum_dof {} accum_snr2 {}".format(accum_dof, accum_snr2)
                if (accum_dof > 0.0) and (accum_snr2) > self.snr_threshold**2:
                    print "accum n {}, accum dof {: 5.2f} accum snr {: 5.2f}".format(len(group_ixs), accum_dof, np.sqrt(accum_snr2))
                    #this is a good enough group add it to the set of groups
                    all_fit_groups.append(group_ixs)
                    #set their fit group column to match each other
                    for feat_ix in group_ixs:
                        self.fdat["fit_group"].ix[feat_ix] = len(all_fit_groups)
                        used_features.add(feat_ix)
                    #subtract from our total allotment of signal to noise and degrees of freedom
                    for min_idx, max_idx, snr2_prod, dof_prod in to_subtract:
                        resid_snr2 = snr2_available[min_idx:max_idx+1] - snr2_prod
                        snr2_available[min_idx:max_idx+1] = np.where(resid_snr2 > 0, resid_snr2, 0.0)
                        resid_dof = dof_available[min_idx:max_idx+1] - dof_prod*min(1.0, 3.0/len(to_subtract))
                        dof_available[min_idx:max_idx+1] = np.where(resid_dof > 0, resid_dof, 0.0)
                else:
                    #no allowable groups for this feature relegate it to the background
                    self.fdat.iloc[feature_idx]["fit_group"] = 0
                feature_idx += 1
        #import pdb; pdb.set_trace()
    
    def get_index(self, wv, clip=False):
        idx = np.log(np.asarray(wv)/self.min_wv)/self.delta_log
        if clip:
            idx = np.clip(idx, 0, self.npts-1)
        return idx
    
    def generate_feature_matrix(self):
        group_gb = self.fdat.groupby("fit_group")
        groups = group_gb.groups
        
        n_groups = len(groups)
        
        self.feature_matrix = scipy.sparse.lil_matrix((self.npts, n_groups), dtype=float)
        group_keys = groups.keys()
        self.fdat["ew_frac"] = np.zeros(len(self.fdat))
        for group_idx in range(len(group_keys)):
            group_id = group_keys[group_idx]
            group_ldf = self.fdat.ix[groups[group_id]]
            group_cog_lrw = group_ldf.cog_lrw.values
            max_lrw = np.max(group_cog_lrw)
            relative_ews = np.power(10, group_cog_lrw-max_lrw)*group_ldf.wv.values
            relative_ew_sum = np.sum(relative_ews)
            ew_fracs = relative_ews/relative_ew_sum
            self.fdat["ew_frac"].ix[groups[group_id]] = ew_fracs
            for feat_idx in range(len(group_ldf)):
                feat_wv = group_ldf.iloc[feat_idx]["wv"]
                wv_idx = self.get_index(feat_wv)
                if wv_idx < 0:
                    continue
                if wv_idx > self.npts-1:
                    continue
                target_width = np.sqrt(group_ldf["doppler_width"].iloc[feat_idx]**2 + group_ldf["sigma_offset"].iloc[feat_idx]**2)
                
                lb = int(np.around(self.get_index(feat_wv-5.0*target_width, clip=True)))
                ub = int(np.around(self.get_index(feat_wv+5.0*target_width, clip=True)))
                
                profile = voigt(self.model_wv[lb:ub+1], feat_wv, target_width, 0.0)
                profile *= ew_fracs[feat_idx]
                
                self.feature_matrix[lb:ub+1, group_idx] = profile.reshape((-1, 1)) + self.feature_matrix[lb:ub+1, group_idx]
        self.feature_matrix = self.feature_matrix.tocsr()
    
    @property
    def theta(self):
        return 5040.0/self.teff
    
    def calc_solar_ab(self):
        self.fdat["solar_ab"] = ptable[self.species.values]["abundance"]
    
    @property
    def solar_ab(self):
        return self.fdat["solar_ab"]
    
    def calc_therm_widths(self):
        #TODO: use real atomic weight
        weight =2.0*self.fdat["Z"]
        wv = self.fdat["wv"]
        widths = 4.301e-7*np.sqrt(self.teff/weight)*wv
        self.fdat["thermal_width"] = widths
    
    def calc_vmicro_widths(self):
        self.fdat["vmicro_width"] = self.fdat["wv"]*self.vmicro/299792.458
    
    def calc_doppler_widths(self):
        self.calc_therm_widths()
        self.calc_vmicro_widths()
        dop_widths = np.sqrt(self.fdat["vmicro_width"]**2 + self.fdat["thermal_width"]**2)
        self.fdat["doppler_width"] = dop_widths
    
    def calc_x(self):
        self.fdat["x"] = self.solar_ab + self.loggf - self.ep*self.theta - self.doppler_lrw
    
    def calc_cog_lrws(self):
        lrws = self.cog(self.x_adj.values) #adjusted?
        self.fdat["cog_lrw_adj"] = lrws
        self.fdat["cog_lrw"] = lrws + self.doppler_lrw.values
    
    def calc_cog(self):
        """calculate an approximate curve of growth and store its representation
        """
        min_log_strength = -1.5
        max_log_strength = 4.5
        n_strength = 100
        log_strengths = np.linspace(min_log_strength, max_log_strength, n_strength)
        strengths = np.power(10, log_strengths)
        npts = 1000
        dx = np.zeros(npts)
        extra_ends = [55, 65, 75, 100, 150, 300, 500, 1000]
        n_ends = len(extra_ends)
        dx[n_ends:-n_ends] = np.linspace(-50, 50, npts-2*n_ends)
        for end_idx in range(n_ends):
            dx[end_idx] = -extra_ends[-(end_idx+1)]
            dx[-(end_idx+1)] = extra_ends[-(end_idx+1)]
        
        opac_profile = voigt(dx, 0.0, 1.0, self.gamma_ratio_5000)        
        log_rews = np.zeros(n_strength)
        for strength_idx in range(n_strength):
            flux_deltas = 1.0-np.exp(-strengths[strength_idx]*opac_profile)
            cur_rew = integrate.trapz(flux_deltas, x=dx)
            log_rews[strength_idx] = np.log10(cur_rew)
        
        cog_slope = scipy.gradient(log_rews)/scipy.gradient(log_strengths)
        
        n_extend = 2
        low_strengths = min_log_strength - np.linspace(0.2, 0.1, n_extend)
        high_strengths = max_log_strength + np.linspace(0.1, 0.2, n_extend)
        extended_log_strengths = np.hstack((low_strengths, log_strengths, high_strengths))
        extended_slopes = np.hstack((np.ones(n_extend), cog_slope, 0.5*np.ones(n_extend)))
        
        cpoints = np.linspace(min_log_strength, max_log_strength, 10)
        #constrained_ppol = ppol.RCPPB(poly_order=1, control_points=cpoints)
        #linear_basis = constrained_ppol.get_basis(extended_log_strengths)
        linear_ppol = ppol.fit_piecewise_polynomial(extended_log_strengths, extended_slopes, order=1, control_points=cpoints)
        
        fit_quad = linear_ppol.integ()
        #set the integration constant and redo the integration
        zero_cross_idx = np.argmin(np.abs(log_rews))
        zero_cross_lstr = log_strengths[zero_cross_idx]
        new_offset = fit_quad([zero_cross_lstr])[0]
        fit_quad = linear_ppol.integ(-new_offset)
        
        #fit_quad = smooth_ppol_fit(log_strengths, log_rews, y_inv=10.0*np.ones(n_strength), order=2)
        self.cog = ppol.InvertiblePiecewiseQuadratic(fit_quad.coefficients, fit_quad.control_points, centers=fit_quad.centers, scales=fit_quad.scales)
    
    def fit_offsets(self):
        non_bk = self.fdat[self.fdat.fit_group > 0]
        bk = self.fdat[self.fdat.fit_group < 0]
        gb_cols = ["Z", "ion"]
        species_gb = non_bk.groupby(gb_cols)
        bk_gb = bk.groupby(gb_cols)
        groups = species_gb.groups
        bk_groups = bk_gb.groups
        for species_key in groups.keys():
            species_df = self.fdat.ix[groups[species_key]]
            delta_rews = np.log10(species_df.ew/species_df.doppler_width)
            x_deltas = species_df.x.values - self.cog.inverse(delta_rews.values)
            offset = np.median(x_deltas)
            self.fdat["x_offset"][groups[species_key]] = offset
            bk_ixs = bk_groups.get(species_key)
            if not bk_ixs is None:
                self.fdat["x_offset"][bk_ixs] = offset
    
    @property
    def doppler_lrw(self):
        return np.log10(self.doppler_width/self.wv)
    
    @property
    def lrw(self):
        return np.log10(self.fdat["ew"]/self.fdat["wv"])
    
    @property
    def x_adj(self):
        return self.x - self.x_offset
    
    @property
    def sigma(self):
        sig = np.sqrt(self.fdat["vmicro_width"]**2 
                + self.fdat["thermal_width"]**2 
                + (self.fdat["wv"]*self.vmacro/299792.458)**2) 
        return sig + self.fdat["sigma_off"]
    
    @property
    def lrw_adj(self):
        return self.lrw - self.doppler_lrw
        
    def cog_plot(self, ax=None):
        foreground = self.fdat[self.fdat.fit_group > 0]
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(foreground.x-foreground.x_offset, np.log10(foreground.ew/foreground.wv))
        ax.set_xlabel("adjusted relative strength")
        ax.set_ylabel("log(EW/doppler_width)")


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
    
