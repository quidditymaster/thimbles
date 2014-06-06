

class Species(object):

    def __init__(self, isotopes):
        self.isotopes = isotopes
    
    def __eq__(self, other):
        try:
            assert len(self.isotopes) == len(other.isotopes) 
            for iso_idx in range(len(self.isotopes)):
                assert self.isotopes[iso_idx] == other.isotopes[iso_idx]
        except AssertionError:
            return False
        return True

class Ionization(object):
    
    def __init__(self, charge):
        self.charge = charge

class Isotope(object):
    
    def __init__(self, element, mass_number, mass):
        self.element = element
        self.mass_number = mass_number
        self.mass = mass

class Transition(object):
    """
    an optically induced transition between energy levels
    
    Raises
    ------
    none
    
    
    Notes
    -----
    __1)__ none
    
    
    Examples
    --------
    """
    
    def __init__ (self, wv, species, ion, ep, loggf):
        self.wv=wv
        self.species=species
        self.ion=ion
        self.loggf=loggf
        self.ep=ep
    
    def __repr__ (self):
        out = (format(self.wv,'10.3f'),
               format(self.species,'5.1'),
               format(self.ep,'5.2f'),
               format(self.loggf,'5.2f'))
        return "  ".join(out)