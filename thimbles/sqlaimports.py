import sqlalchemy as sa
from sqlalchemy import ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float
from sqlalchemy import PickleType, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref

class PolymorphicTable(object):

    @declared_attr
    def __mapper_args__(self):
        pass
