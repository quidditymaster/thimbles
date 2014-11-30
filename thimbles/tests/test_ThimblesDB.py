import unittest
import thimbles as tmb
from thimbles.thimblesdb import ThimblesTable, Base, ThimblesDB
from thimbles.sqlaimports import *

import os
import numpy as np

class Dummy(ThimblesTable, Base):
    name  = Column(String)
    float_value = Column(Float)
    int_value = Column(Integer)
    array_value = Column(PickleType)
    
    def __init__(self, name=None, float_value=None, int_value=None, array_value=None):
        self.name = name
        self.float_value = float_value
        self.int_value = int_value
        self.array_value=array_value

class TestCreate(unittest.TestCase):
    
    def setUp(self):
        base_path = os.path.abspath(os.path.dirname(__file__))
        self.db_path = os.path.join(base_path, "junk_db.tdb")
    
    def delete_db(self):
        os.system( "rm -r {}".format(self.db_path))
    
    def test_create(self):
        #make from scratch
        self.delete_db()
        tdb = ThimblesDB(self.db_path)
        self.assertTrue(os.path.exists(self.db_path))

class TestPersistence(unittest.TestCase):
    
    def setUp(self):
        base_path = os.path.abspath(os.path.dirname(__file__))
        self.db_path = os.path.join(base_path, "junk_db.tdb")
        self.delete_db()
        self.make_db()
    
    def make_db(self):
        self.tdb = ThimblesDB(self.db_path)
    
    def delete_db(self):
        os.system( "rm -r {}".format(self.db_path))
    
    def test_save_dummy(self):
        dummy1 = Dummy(name="groot", float_value=60.2, int_value=89)
        dummy2 = Dummy(name="bloot", float_value=20.2, int_value=2, array_value=np.arange(10))
        self.tdb.add(dummy1)
        self.tdb.add(dummy2)
        self.tdb.save()
        res = self.tdb.query(Dummy).all()
        print res
        self.assertTrue(len(res) == 2)
        d1_res = self.tdb.query(Dummy).filter(Dummy.name == "groot").first()
        self.assertTrue(d1_res is dummy1)
        self.tdb.close()
        self.make_db()
        afres = self.tdb.query(Dummy).filter(Dummy.name=="groot").first()
        self.assertTrue(afres.int_value == 89)
        self.assertTrue(id(afres) != id(dummy1))

if __name__ == "__main__":
    unittest.main()
