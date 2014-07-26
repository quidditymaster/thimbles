from thimbles.thimblesdb import Base, ThimblesTable

from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref

class Spectrograph(Base, ThimblesTable):
    name = Column(String)

class SpectrographSetup(Base, ThimblesTable):
    spectrograph_id = Column(Integer, ForeignKey("Spectrograph._id"))
    spectrograph = relationship(Spectrograph)
