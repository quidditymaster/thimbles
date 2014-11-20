from thimbles.abundances import AbundanceVector

class StellarParameters(object):
    
    def __init__(self, teff, logg, abundances=0.0, vmicro=2.0):
        self.teff = teff
        self.logg = logg
        if abundances is None:
            abundances = AbundanceVector(0.0)
        self.abundances = abundances
        self.vmicro = vmicro
    
    @property
    def theta(self):
        return 5040/self.teff

solar_parameters = StellarParameters(5777.0, 4.44, abundances=0.0, vmicro=0.88)

