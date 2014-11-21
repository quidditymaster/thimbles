from copy import deepcopy
import os
#from thimbles.sqlaimports import *
from thimbles import Session
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy import create_engine, Column, Integer
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
    
    def __init__(self, path, create=False):
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            if create:
                os.makedirs(self.path)
                #give the file system a moment to catch up.
                import time;time.sleep(0.02)
            else:
                raise IOError("path does not exisit")
        
        #set up the database
        self.db_url = "sqlite:////{}".format(self.path)
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.session = Session()
    
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

