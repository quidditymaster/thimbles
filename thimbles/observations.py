
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.tasks import task
from thimbles.sqlaimports import *
from thimbles.modeling.associations import HasParameterContext

class Observation(Base, ThimblesTable, HasParameterContext):
    start = Column(DateTime)
    duration = Column(Float) #in seconds
    airmass = Column(Float)
    observation_type = Column(String)
    __mapper_args__={
        "polymorphic_on":observation_type,
        "polymorphic_identity":"Observation"
    }
    
    def __init__(self, start=None, duration=None, airmass=None):
        HasParameterContext.__init__(self)
        self.start = start
        self.duration = duration
        self.airmass = airmass

    
def prefer_existing(spec, database, matches):
    return matches[0]

def obs_from_spec(spec, database):
    Observation

@task()
def update_observations(
        spectra, 
        obs_matcher, 
        on_matched, 
        on_matchless
):
    pass
    
