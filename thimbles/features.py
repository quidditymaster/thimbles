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
from latbin import matching

class SaturatedVoigtFeatureModel(object):
    
    def __init__(self, 
                 transitions, 
                 min_wv,
                 max_wv,
                 teff=5000.0, 
                 vmicro=2.0, 
                 vmacro=1.0, 
                 gamma_ratio_5000=0.02,
                 initial_x_offset=4.0,
                 delta_wv_max=100,
                 delta_x_max=0.5,
                 domination_ratio=4.0,
                 model_resolution=2e5,):
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
        self.min_wv = min_wv
        self.max_wv = max_wv
        self.teff = teff
        self._initial_x_off = initial_x_offset
        self.vmicro = vmicro
        self.vmacro = vmacro
        self.domination_ratio
        self.delta_x_max
        self.delta_wv_max
        #TODO: make the gamma ratio change as a function of wv
        #TODO: allow for individual gamma ratio's by interpolating cogs
        self.gamma_ratio_5000 = gamma_ratio_5000
        self.model_resolution=model_resolution
        
        self.n_pts = int(np.log(self.max_wv/self.min_wv)*self.model_resolution)
        self.model_wv = np.exp(np.linspace(np.log(self.min_wv), np.log(self.max_wv), self.n_pts))
        self.model_wv_gradient = scipy.gradient(self.model_wv)
        self.delta_log_wv = np.log(self.max_wv/self.min_wv)/self.n_pts
        
        self.fdat = transitions.copy()
        self.n_feat = len(self.fdat)
        self._initialize_columns()
        #a is defined as gamma*wv**2/(4*pi*c*doppler_width)
    
    def _initialize_columns(self):
        cur_columns = self.fdat.columns 
        targ_columns = ["ew", "wv_off", "sigma_off", "gamma_off"]
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
        
    def optimize_fit_parameters(self, spectra):
        transforms_to_model = []
        npts_model = len(self.model_wv)
        min_delta_pix = np.empty(npts_model, dtype=float)
        min_delta_pix.fill(1000.0) #bigger than any reasonable pixel size
        snr2_available = np.zeros(npts_model)
        dof_available = np.zeros(npts_model)
        
        verbosity("building spectra to model space transforms")
        for spec in spectra:
            trans = tmb.utils.resampling.get_resampling_matrix(spec.wv, self.model_wv, preserve_normalization=False)
            transforms_to_model.append(trans)
            
            cur_snr2 = trans*(spec.flux**2*spec.inv_var)
            cur_dof = trans*np.ones(spec.wv.shape)
            cur_delta_pix = trans*scipy.gradient(spec.wv)
            cur_delta_pix = np.where(cur_delta_pix > 0, cur_delta_pix, 1000.0)
            min_delta_pix = np.where(cur_delta_pix < min_delta_pix, cur_delta_pix, min_delta_pix)
            snr2_available += cur_snr2
            dof_available = np.where(dof_available > cur_dof, dof_available, cur_dof)
        
        #get the log of the min pixel size as a fraction of wavelength
        pixel_lrw = np.log10(min_delta_pix/self.model_wv)
        
        #admit features up to 1/100 of a pixel width in predicted equivalent width
        max_strengths = np.power(10.0, pixel_lrw - 2.0) 
        
        window_delta = 0.001
        verbosity("collecting max feature strengths per model pixel")
        for line_idx in range(len(self.fdat)):
            ldata = self.fdat.iloc[line_idx]
            cent_wv = ldata.wv
            min_idx = self.get_index(ldata.wv*(1.0-window_delta))
            max_idx = self.get_index(ldata.wv*(1.0+window_delta))
            sigma = ldata["doppler_width"]
            #TODO: replace the constant gamma with an appropriately varying one
            gamma = self.gamma_ratio_5000*sigma
            prof = voigt(self.model_wv[min_idx:max_idx+1], cent_wv, sigma, gamma)
            predicted_lrw = ldata["cog_lrw"]
            prof *= np.power(10.0, predicted_lrw)/np.max(prof)
            mst_snippet = max_strengths[min_idx:max_idx+1]
            max_strengths[min_idx:max_idx] = np.where(mst_snippet > prof, mst_snippet, prof)
        
        verbosity("relegating dominated features to a background model")
        #set up the fit_type column
        self.fdat["is_background"] = np.zeros()
        for line_idx in range(len):
            ldata = self.fdat.iloc[line_idx]
            cent_wv = ldata.wv
            cent_idx = self.get_index(cent_wv)
            predicted_lrw = ldata["cog_lrw"]
            if max_strengths[cent_idx] > self.domination_ratio*np.power(10.0, predicted_lrw):
                self.fdat.iloc[line_idx, "is_background"] = 1
        
        verbosity("matching potential groups")
        foreground_features = self.fdat[self.fdat.fit_type == 1]
        
        species_gb = foreground_features.groupby(["Z", "ion"])
        for all_idxs in species_gb.groups:
            species_df = foreground_features.ix[all_idxs]
            last_grouping = None
            wv_x_vals = species_df[["wv", "x"]].values.copy()
            wv_x_vals[:, 0]/self.delta_wv_max
            wv_x_vals[:, 1]/self.delta_x_max
            m1, m2, dist = matching.match(wv_x_vals, wv_x_vals, tolerance=1.0) 
            match_idx = 0
            feature_idx = 0
            used_features = set()
            while feature_idx < len(species_df):
                if feature_idx in used_features:
                    feature_idx += 1
                    continue
                potential_matches = []
                while (m1[match_idx] == feature_idx) and (match_idx < len(m1)):
                    if m2[match_idx] != feature_idx:
                        potential_matches.append(m2[match_idx])
                    match_idx += 1 
                
    
    def get_index(self, wv, clip=False):
        idx = np.log(np.asarray(wv)/self.min_wv)/self.delta_log
        if clip:
            np.clip(idx, 0, self.npts-1, out=idx)
        return idx
    
    def generate_feature_matrix(self):
        group_gb = self.fdat.groupby("group_id")
        groups = group_gb.groups
        
        n_groups = len(groups)
        
        self.feature_matrix = scipy.sparse.lil_matrix((self.npts, n_groups), dtype=float)
        group_keys = groups.keys()
        for group_idx in range(len(group_keys)):
            group_id = group_keys[group_idx]
            group_ldf = self.fdat.ix[groups[group_id]]
            group_cog_lrw_adj = group_ldf.cog_lrw.values
            max_lrw = np.max(group_cog_lrw_adj)
            relative_ews = np.power(10, max_lrw-group_cog_lrw_adj)*group_ldf.wv.values
            relative_ew_sum = np.sum(relative_ews)
            print group_id
            for feat_idx in range(len(group_ldf)):
                feat_wv = group_ldf.iloc[feat_idx]["wv"]
                wv_idx = self.get_index(feat_wv)
                if wv_idx < 0:
                    continue
                if wv_idx > self.npts-1:
                    continue
                thermal_width = group_ldf.iloc[feat_idx]["therm_width"]
                pix_width = thermal_width/self.wv_gradient[int(wv_idx)]
                lb = max(0, int(wv_idx+pix_width*5))
                ub = min(self.npts-1, int(wv_idx+pix_width*5)+1)
                pix_dx = np.arange(lb, ub)-wv_idx
                
                profile = voigt(pix_dx, 0.0, pix_width, 0.0)
                profile /= np.sum(profile)
                profile *= relative_ews[feat_idx]/relative_ew_sum
                
                self.feature_matrix[lb:ub, group_idx] += profile
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
        self.fdat["cog_lrw_true"] = lrws + self.doppler_lrw.values
    
    def calc_cog(self):
        """calculate an approximate curve of growth and store its representation
        """
        n_strength = 100
        strengths = np.power(10, np.linspace(-1.0, 5.0, n_strength))
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
        
        fit_quad = smooth_ppol_fit(np.log10(strengths), log_rews, y_inv=10.0*np.ones(n_strength), order=2)
        self.cog = ppol.InvertiblePiecewiseQuadratic(fit_quad.coefficients, fit_quad.control_points)
    
    def fit_offsets(self):
        species_gb = self.fdat.groupby(["Z", "ion"])
        groups = species_gb.groups
        for species_key in groups.keys():
            species_df = self.fdat.ix[groups[species_key]]
            delta_rews = np.log10(species_df.ew/species_df.doppler_width)
            x_deltas = species_df.x.values - self.cog.inverse(delta_rews.values)
            offset = np.median(x_deltas)
            self.fdat["x_offset"][groups[species_key]] = offset
        
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
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.x_adj, self.lrw_adj)
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
    
