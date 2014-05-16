import numpy as np

class InertialFrame(object):
    """an inertial reference frame, decomposed on the plane of the sky
    with the earth sun barycenter as v=0. Positive velocities are away
    from the earth sun system.
    """
    
    def __init__(self, barycenter_rv, pm_ra=None, pm_dec=None):
        self.rv = barycenter_rv
        self.pm_ra = pm_ra
        self.pm_dec = pm_dec
    
    def __sub__(self, other):
        if isinstance(other, InertialFrame):
            other_rv = other.rv
        else:
            other_rv = float(other)
        new_rv = self.rv - other_rv
        return InertialFrame(new_rv)

    def __add__(self, other):
        if isinstance(other, InertialFrame):
            other_rv = other.rv
        else:
            other_rv = float(other)
        new_rv = self.rv + other_rv
        return InertialFrame(new_rv)
