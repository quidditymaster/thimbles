
from .db.dbclasses import Base, ThimblesTable

from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, DateTime, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref

class SpectrumObservation(Base, ThimblesTable):
    spectrum_id = Column(Integer, ForeignKey("Source._id"))
    setup_id = Column(Integer, ForeignKey("SpectrographSetup._id"))
    setup = relationship("SpectrographSetup")
    start = Column(DateTime)
    duration = Column(Float) #in seconds


