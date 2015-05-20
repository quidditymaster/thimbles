from thimbles.thimblesdb import Base, ThimblesTable
from .sqlaimports import *

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
    info = Column(PickleType)
    
    def __init__(self, name=None, ra=None, dec=None, info=None):
        self.name = name
        self.ra=ra
        self.dec=dec
        if info is None:
            info = {}
        self.info = info

grouping_assoc = sa.Table(
    "source_group_assoc", 
    Base.metadata,
    Column("source_id", Integer, ForeignKey("Source._id")),
    Column("group_id", Integer, ForeignKey("SourceGroup._id")),
)

class SourceGroup(Base, ThimblesTable):
    name = Column(String)
    sources = relationship("Source", secondary=grouping_assoc)
    
    def __init__(self, name="", sources=None):
        self.name = name
        if sources is None:
            sources = []
        self.sources = sources

