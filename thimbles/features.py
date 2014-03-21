import numpy as np
#from .stellar_atmospheres import periodic_table
import thimbles as tmb

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

class FeatureFlags(object):
    
    def __init__(self):
        self.flags = {}
        self.flags["use"] = True
        self.flags["in_range"] = False
        self.flags["bad_data"] = False
        self.flags["bad_fit"]  = False
        self.flags["viewed"] = False
    
    def set_true(self, *flag_names):
        if not len(flag_names):
            flag_names = self.flags.keys()
        
        for flag_name in flag_names:
            self.flags[flag_name] = True
    
    def set_false(self, *flag_names):
        if not len(flag_names):
            flag_names = self.flags.keys()
        for flag_name in flag_names:
            self.flags[flag_name] = False
    
    def __getitem__(self, index):
        return self.flags[index]
    
    def __setitem__(self, index, value):
        self.flags[index] = value

class Feature(object):
    
    def __init__(self, 
                 profile, 
                 eq_width, 
                 abundance, 
                 trans_parameters,
                 relative_continuum=1.0, 
                 data_sample=None,
                 flags=None):
        self.profile = profile
        self._eq_width = eq_width
        self.abundance = abundance
        self.trans_parameters = trans_parameters
        self.data_sample=data_sample
        if flags == None:
            flags = FeatureFlags()
        self.flags = flags
        self.relative_continuum=relative_continuum
    
    def __repr__ (self):
        return "Feature : "+repr(self.trans_parameters)
    
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
