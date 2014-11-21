from copy import deepcopy
import os
from thimbles.sqlaimports import *
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy import create_engine
from thimbles.options import Option, opts, OptionSpecificationError
from sqlalchemy.orm import sessionmaker
Session = sessionmaker()

Base = declarative_base()

class ThimblesTable(object):
    _id = Column(Integer, primary_key=True)    
    
    @declared_attr
    def __tablename__(cls):
        return cls.__name__


class ThimblesDB(object):
    """encapsulates 
    """
    
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.db_url = "sqlite:///{}".format(self.path)
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.session = Session(bind=self.engine)
    
    def add(self, obj):
        self.session.add(obj)
    
    def add_all(self, obj_list):
        self.session.add_all(obj_list)
    
    def query(self, *args, **kwargs):
        return self.session.query(*args, **kwargs)
    
    def commit(self):
        self.session.commit()
    
    def close(self):
        self.session.close()

