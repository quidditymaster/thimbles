
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling.associations import HasParameterContext, NamedParameters, ParameterAliasMixin
from .sqlaimports import *

class SourceAlias(ParameterAliasMixin, ThimblesTable, Base):
    _context_id = Column(Integer, ForeignKey("Source._id"))
    context= relationship("Source", foreign_keys=_context_id, back_populates="context")

class Source(Base, ThimblesTable, HasParameterContext):
    """astrophysical light source"""
    source_class = Column(String)
    context = relationship("SourceAlias", collection_class=NamedParameters)
    __mapper_args__={
        "polymorphic_identity":"Source",
        "polymorphic_on":source_class,
    }
    name = Column(String)
    ra = Column(Float)
    dec = Column(Float)
    info = Column(PickleType)
    spectroscopy = relationship("Spectrum")
#    spectroscopy = relationship(
#        "Spectrum", 
#        primaryjoin="and_(Source._id==remote(Observation._source_id), foreign(Spectrum._observation_id)==Observation._id)",
#        viewonly=True,
#    )
    photometry = relationship(
        "Photom",
    )
    
    def __init__(self, name=None, ra=None, dec=None, info=None):
        HasParameterContext.__init__(self)
        self.name = name
        self.ra=ra
        self.dec=dec
        if info is None:
            info = {}
        self.info = info
    
    def __repr__(self):
        return "<Source: {}>".format(self.name)
    
    def add_parameter(self, name, parameter, is_compound=False):
        SourceAlias(name=name, context=self, parameter=parameter, is_compound=is_compound)


###
class GroupingAlias(ParameterAliasMixin, ThimblesTable, Base):
    _context_id = Column(Integer, ForeignKey("SourceGrouping._id"))
    context= relationship("SourceGrouping", foreign_keys=_context_id, back_populates="context")


grouping_assoc = sa.Table(
    "source_grouping_assoc", 
    Base.metadata,
    Column("source_id", Integer, ForeignKey("Source._id")),
    Column("group_id", Integer, ForeignKey("SourceGrouping._id")),
)

class SourceGrouping(Base, ThimblesTable, HasParameterContext):
    name = Column(String)
    sources = relationship("Source", secondary=grouping_assoc)
    context = relationship("GroupingAlias", collection_class=NamedParameters)
    info = Column(PickleType)
    
    def __init__(self, name=None, sources=None, info=None):
        self.name = name
        if sources is None:
            sources = []
        self.sources = sources
        if info is None:
            info = {}
        self.info = info
    
    def add_parameter(self, name, parameter, is_compound=False):
        GroupingAlias(name=name, context=self, parameter=parameter, is_compound=is_compound)
