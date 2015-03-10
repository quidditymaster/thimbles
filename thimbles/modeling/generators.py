import thimbles as tmb
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.sqlaimports import *

class ModelSubstrate(ThimblesTable, Base):
    #.models attribute as a backref from Model
    substrate_class = Column(String)
    __mapper_args__={
        "polymorphic_identity":"ModelSubstrate",
        "polymorphic_on":substrate_class,
    }
    
