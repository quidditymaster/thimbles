from thimbles.thimblesdb import Base, ThimblesTable
from sqlaimports import *

class Source(Base, ThimblesTable):
    """astrophysical light source"""
    source_class = Column(String)
    __mapper_args__={
        "polymorphic_identity":"Source",
        "polymorphic_on":source_class,
    }
    name = Column(String)
    ra = Column(Float)
    dec = Column(Float)
    
    def __init__(self, name=None, ra=None, dec=None):
        self.name = name
        self.ra=ra
        self.dec=dec
    
