from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.sqlaimports import *
from thimbles.modeling import Parameter

class Photom(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"Photom"
    }
    filter_id = Column(Integer, ForeignKey("Filter._id"))
    filter = relationship("Filter")
    source_id = Column(Integer, ForeignKey("Source._id"))
    source = relationship("Source")
    
    def __init__(self, source, value, filter):
        self.source = source
        self._value = value
        self.filter = filter


class Filter(Base, ThimblesTable):
    name = Column(String, unique=True)

