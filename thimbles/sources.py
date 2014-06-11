
from .db.dbclasses import Base, ThimblesTable

from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref

class Source(Base, ThimblesTable):
    """the observed astrophysical source"""
    name = Column(String)
    ra = Column(Float)
    dec = Column(Float)
    photometry = relationship("Photometry")
    observations = relationship("SpectrumObservation")

