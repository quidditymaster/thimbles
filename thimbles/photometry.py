from .db.dbclasses import Base, ThimblesTable

from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref

class Photometry(Base, ThimblesTable):
    filter_id = Column(Integer, ForeignKey("Filter._id"))
    filter = relationship("Filter")
    source_id = Column(Integer, ForeignKey("Source._id"))
    magnitude = Column(Float)
    error = Column(Float)

class Filter(Base, ThimblesTable):
    name = Column(String)