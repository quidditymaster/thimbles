
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.tasks import task
from thimbles.sqlaimports import *
from thimbles.modeling.associations import HasParameterContext, NamedParameters, ParameterAliasMixin

class PointingAlias(ParameterAliasMixin, ThimblesTable, Base):
    _context_id = Column(Integer, ForeignKey("Pointing._id"))
    context= relationship("Pointing", foreign_keys=_context_id, back_populates="context")

class Pointing(Base, ThimblesTable, HasParameterContext):
    context = relationship("PointingAlias", collection_class=NamedParameters)
    ra = Column(Float)
    dec = Column(Float)
    tai = Column(Float) #seconds
    duration = Column(Float) #seconds
    airmass = Column(Float)
    pointing_type = Column(String)
    __mapper_args__={
        "polymorphic_on":pointing_type,
        "polymorphic_identity":"Pointing"
    }
    
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
    
    def add_parameter(name, parameter, is_compound=False):
        PointingAlias(name=name, context=self, parameter=parameter, is_compound=is_compound)


class ObservationAlias(ParameterAliasMixin, ThimblesTable, Base):
    _context_id = Column(Integer, ForeignKey("Observation._id"))
    context= relationship("Observation", foreign_keys=_context_id, back_populates="context")

class Observation(Base, ThimblesTable, HasParameterContext):
    context = relationship("ObservationAlias", collection_class=NamedParameters)
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
    
    def add_parameter(name, parameter, is_compound=False):
        ObservationAlias(name=name, context=self, parameter=parameter, is_compound=is_compound)
