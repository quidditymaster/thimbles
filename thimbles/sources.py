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
    #photometry = relationship("")
    spectroscopy = relationship("Spectrum", backref="source")
    
    def __init__(self, name=None, ra=None, dec=None, photometry=None, spectroscopy=None):
        self.name = name
        self.ra=ra
        self.dec=dec
        if photometry is None:
            photometry = []
        self.photometry=photometry
        if spectroscopy is None:
            spectroscopy = []
        self.spectroscopy = spectroscopy
