
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.sqlaimports import *

class Observation(Base, ThimblesTable):
    start = Column(DateTime)
    duration = Column(Float) #in seconds
    airmass = Column(Float)
