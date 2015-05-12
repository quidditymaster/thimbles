from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.sqlaimports import *
from thimbles.modeling import Parameter

class Photometry(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    filter_id = Column(Integer, ForeignKey("Filter._id"))
    filter = relationship("Filter")
    source_id = Column(Integer, ForeignKey("Source._id"))
    source = relationship("Source")
    
    def __init__(self, filter, source, value):
        self.filter = filter
        self.source = source
        self._value = value


class Filter(Base, ThimblesTable):
    name = Column(String, unique=True)

