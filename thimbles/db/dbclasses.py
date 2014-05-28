
import numpy as np

from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref

#engine = create_engine('sqlite://', echo=True)
from .thimblesdb import engine
Base = declarative_base()

class ThimblesTable(object):
    
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()[1:]
    
    _id = Column(Integer, primary_key=True)
    note = Column(String)

class DSourceObject(Base, ThimblesTable):
    """the observed astrophysical source"""
    
    name = Column(String)
    ra = Column(Float)
    dec = Column(Float)
    
    def __init__(self, name="", ra=None, dec=None):
        self.name = name
        self.ra = ra
        self.dec = dec

class DAtomicDataSource(Base, ThimblesTable):
    """the provenance of pieces of data"""
    
    name = Column(String)
    priority = Column(Integer)
    reference = Column(String)

class DLogGF(Base, ThimblesTable):
    loggf = Column(Float)
    #reference = relationship("DAtomicDataSource", backref=backref(__tablename__, order_by=_id))


Base.metadata.create_all(engine)
