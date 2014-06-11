import os

import numpy as np

from sqlalchemy import ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
current_dbs = {}

def get_db(path):
    abspath = os.path.abspath(path)
    new_db = current_dbs.get(abspath)
    if not new_db is None:
        return new_db
    else:
        return(ThimblesDB(abspath))

class ThimblesTable(object):
    
    @declared_attr
    def __tablename__(cls):
        return cls.__name__
    
    _id = Column(Integer, primary_key=True)


class ThimblesDB(object):
    """ a class to encapsulate a thimbles style data format as a combination
    of an SQL database an hdf5 file and pickle files. (SQL to handle irregular 
    collections and links between data, hdf5 to handle large regular
    numerical data arrays and pickle to handle everything else.)
    """
    
    def __init__(self, path):
        self.db_url = "sqlite:///{}".format(os.path.abspath(path))
        
        self.engine = create_engine(self.db_url)
        #engine = create_engine("sqlite:////home/tim/sandbox/sqlalchemy/test.db")
        
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
