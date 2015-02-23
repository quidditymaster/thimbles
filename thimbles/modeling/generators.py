import thimbles as tmb
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.sqlaimports import *

class Generator(ThimblesTable, Base):
    #.models attribute as a backref from Model
    
    @classmethod
    def request(cls, targets, database):
        pass

