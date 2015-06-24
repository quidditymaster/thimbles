
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.tasks import task
from thimbles.sqlaimports import *
from thimbles.modeling.associations import HasParameterContext

class Pointing(Base, ThimblesTable, HasParameterContext):
    ra = Column(Float)
    dec = Column(Float)
    tai = Column(Float) #seconds
    duration = Column(Float) #seconds
    airmass = Column(Float)
    
    def __init__(
            self, 
            ra=None, 
            dec=None, 
            tai=None, 
            duration=None, 
            airmass=None
    ):
        HasParameterContext.__init__(self)
        self.ra = ra
        self.dec = dec
        self.tai = tai
        self.duration = duration
        self.airmass = airmass


class Observation(Base, ThimblesTable, HasParameterContext):
    _pointing_id = Column(Integer, ForeignKey("Pointing._id"))
    pointing = relationship("Pointing", foreign_keys=_pointing_id)
    _source_id = Column(Integer, ForeignKey("Source._id"))
    source = relationship("Source", foreign_keys=_source_id)
    observation_type = Column(String)
    __mapper_args__={
        "polymorphic_on":observation_type,
        "polymorphic_identity":"Observation"
    }
    
    def __init__(self, pointing, source):
        HasParameterContext.__init__(self)
        self.pointing = pointing
        self.source = source
