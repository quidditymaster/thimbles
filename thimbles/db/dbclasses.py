
import numpy as np

from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref

#engine = create_engine('sqlite://', echo=True)
Base = declarative_base()
import thimbles as tmb

class ThimblesTable(object):
    
    @declared_attr
    def __tablename__(cls):
        return cls.__name__
    
    _id = Column(Integer, primary_key=True)

class Note(object):
    """an sqlalchemy mixin class to provide a notes column"""
    note = Column(String)

class SourceObject(Base, ThimblesTable):
    """the observed astrophysical source"""
    name = Column(String)
    ra = Column(Float)
    dec = Column(Float)
    photometry = relationship("Photometry")

class Photometry(Base, ThimblesTable):
    filter_id = Column(Integer, ForeignKey("Filter._id"))
    filter = relationship(Filter)
    source_id = Column(Integer, ForeignKey("SourceObject._id"))
    magnitude = Column(Float)
    error = Column(Float)

class Filter(Base, ThimblesTable):
    name = Column(String)

class Transition(Base, ThimblesTable):
    pass

#class Provenance(Base, ThimblesTable):
#    """the origin of the data"""
#    name = Column(String)
#    priority = Column(Integer)
#    reference = Column(String)
#
# class Loggf(Base, ThimblesTable):#, TransitionDatum):
#     loggf = Column(Float)
#     tds = relationship("TransitionDataSource")
#     transition_id = Column(Integer, ForeignKey("Transition._id"))
#     #transition_data_id = Column(Integer, ForeignKey("TransitionDataSource._id"))
# 
# class ExcitationPotential(Base, ThimblesTable):#, TransitionDatum):
#     ep = Column(Float)
#     tds = relationship("TransitionDataSource")
#     transition_id = Column(Integer, ForeignKey("Transition._id"))
#     #transition_data_id = Column(Integer, ForeignKey("TransitionDataSource._id"))
# 
# class Wavelength(Base, ThimblesTable):#, TransitionDatum):
#     wv = Column(Float)
# 
# class Blend(Base, ThimblesTable):
#     flags = Float(Integer)
# 
# class EquivalentWidthMeasurement(Base, ThimblesTable):
#     blend_id = Column(Integer, ForeignKey("Blend._id"))
#     user_id = Column(Integer, ForeignKey("User._id"))
#     date = Column(Date)
# 
# class User(Base, ThimblesTable):
#     name = Column(String)
# 
# class Feature(Base, ThimblesTable, tmb.features.Feature):
#     blend_id = Column(Integer, ForeignKey("Blend._id"))
#     transition_id = Column(Integer, ForeignKey("Transition._id"))
#     profile_id = Column(Integer, ForeignKey("Profile._id"))
# 
# class Profile(Base, ThimblesTable,tmb.line_profiles.Voigt):
#     feature_id = Column(Integer, ForeignKey("Feature._id"))
#     sigma = Column(Float)
#     gamma = Column(Float)
# 
# class StellarParameters(Base, ThimblesTable):
#     source_object_id = Column(Integer, ForeignKey("SourceObject._id"))
# 
# #class Spectrum(Base, ThimblesTable, Spectrum):
# #    pass
    
    
