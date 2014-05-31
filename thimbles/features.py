import numpy as np
#from .stellar_atmospheres import periodic_table
import thimbles as tmb
from flags import FeatureFlags
from line_profiles import Gaussian

class Element(object):
    
    def __init__(self, proton_number, symbol, name, reference_isotope):
        self.proton_number = proton_number
        self.symbol = symbol
        self.name = name
        self.reference_isotope = reference_isotope

class Isotope(object):
    
    def __init__(self, element, mass_number, mass):
        self.element = element
        self.mass_number = mass_number
        self.mass = mass

class Species(object):
    
    def __init__(self, isotopes, ionization):
        self.isotopes = isotopes
        self.ionization = ionization

class IonizationStage(object):
    
    def __init__(self, charge):
        self.charge = charge

class Transition(object):
    """
    an optically induced transition between energy levels
    
    wavelength,loggf,ep,vwdamp=0,d0=0
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
    """
    
    def __init__ (self, wv, isotope, loggf, ep, vwdamp=0, d0=0):
        self.wv=wv
        self.iso=isotope
        self.loggf=loggf
        self.ep=ep
        self.vwdamp=vwdamp
        self.d0=d0
    
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

class Feature(object):
    
    def __init__(self,
                 profile,  
                 transition,
                 flags=None,
                 note=""):
        self.profile = profile
        self.transition= transition
        if flags == None:
            flags = FeatureFlags()
        self.flags = flags
        self.note=note
    
    def __repr__ (self):
        rep_str = """Feature : %s notes: %s"""
        return rep_str % (repr(self.trans_parameters), self.notes) 
    
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


class FeatureGroup(object):
    
    def __init__(self, features, spectra, offsets=None, relative_continuum=1.0):
        self.features = features
        self.spectra = spectra
        self.relative_continuum=relative_continuum 
        #offset per spectrum group
        if offsets is None:
            offsets = [0.0 for i in range(spectra)]
        self.offsets = offsets
    
    def get_offset(self, spec_idx=None):
        if spec_idx is None:
            return self.offsets
        return self.offsets[spec_idx]
    
    def set_offset(self, new_off, spec_idx=None):
        if spec_idx is None:
            self.offsets = new_off
        else:
            self.offsets[spec_idx] = new_off
    
    def chi_sq_vec(self, wvs=None):
        if wvs == None:
            wvs = self.data_sample.wv
    
    def trough_bounds(self, wvs, fraction=0.95):
        pass
    
