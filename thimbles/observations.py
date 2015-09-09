
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.tasks import task
from thimbles.sqlaimports import *
from thimbles.modeling.associations import HasParameterContext, NamedParameters, ParameterAliasMixin


class ExposureAlias(ParameterAliasMixin, ThimblesTable, Base):
    _context_id = Column(Integer, ForeignKey("Exposure._id"))
    context= relationship("Exposure", foreign_keys=_context_id, back_populates="context")


class Exposure(Base, ThimblesTable, HasParameterContext):
    context = relationship("ExposureAlias", collection_class=NamedParameters)
    name = Column(String)
    time = Column(Float) #seconds
    duration = Column(Float) #seconds
    type = Column(String)
    info = Column(PickleType)
    
    def __init__(
            self, 
            name,
            time=None, 
            duration=None, 
            type=None,
            info=None,
    ):
        HasParameterContext.__init__(self)
        self.name = name
        self.time = time
        self.duration = duration
        self.type = type
        if info is None:
            info = {}
        self.info = info
    
    def add_parameter(self, name, parameter, is_compound=False):
        ExposureAlias(name=name, context=self, parameter=parameter, is_compound=is_compound)



