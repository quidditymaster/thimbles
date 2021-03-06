#from thimbles.abundances import Abundance
import os
import shutil

class PhotosphereEngine(object):
    """an abstract class specifying the API for wrapping model
    photosphere generating programs"""
    
    def __init__(self):
        pass
    
    def _not_implemented(self):
        raise NotImplementedError("Not implemented for this engine type")
    
    def make_photosphere(self, fname, stellar_params):
        self._not_implemented()
    

class RadiativeTransferEngine(object):
    """a class specifying the API for wrapping radiative transfer
    codes.
    """
    _photosphere_fname = "modelphoto.tmp"
    _last_sparams = None
    
    def __init__(self, working_dir, photosphere_engine=None):
        if not isinstance(working_dir, str):
            raise TypeError("working directory must be a string not type{}".format(type(working_dir)))
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        self.photosphere_engine = photosphere_engine
    
    def _make_photo_file(self, stellar_params):
        if self._last_sparams == stellar_params:
            return
        self._last_sparams = stellar_params
        photo_file = os.path.join(self.working_dir, self._photosphere_fname)
        if isinstance(stellar_params, str):
            if os.path.exists(stellar_params):
                shutil.copy(stellar_params, photo_file)
            else:
                raise IOError("model atmosphere file {} not found".format(stellar_params))
        else:
            self.photosphere_engine.make_photosphere(photo_file, **stellar_params)
    
    def _not_implemented(self, msg=None):
        if msg is None:
            msg = "Not implemented for this engine type"
        raise NotImplementedError(msg)
    
    def ew_to_abundance(self, linelist, stellar_params):
        """generate abundances on the basis of equivalent widths"""
        self._not_implemented()
    
    def line_abundance(self, linelist, stellar_params, inject_as=None):
        self._not_implemented()
    
    def abundance_to_ew(self, linelist, stellar_params, abundances=None):
        self._not_implemented()
    
    def spectrum(self, linelist, stellar_params, wavelengths=None, normalized=True):
        self._not_implemented
    
    def continuum(self, stellar_params):
        self._not_implemented()
    
    def line_strength(self, linelist, stellar_params):
        self._not_implemented()
