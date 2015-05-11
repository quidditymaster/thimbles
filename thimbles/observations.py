
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.sqlaimports import *

class Observation(Base, ThimblesTable):
    start = Column(DateTime)
    duration = Column(Float) #in seconds
    airmass = Column(Float)
    observation_type = Column(String)
    __mapper_args__={
        "polymorphic_on":observation_type,
        "polymorphic_identity":"Observation"
    }
