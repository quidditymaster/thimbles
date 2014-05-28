
import os

#default database path an in memory sqlite database
_db_url = "sqlite://"

def set_db_path (path):
    """a module method to set the path to the thimbles database
    it must be called before db.db is imported e.g.

    import thimlbes.db
    thimbles.db.set_db_path("/path/to/database")
    from thimbles.db import db

    if this function is not called before thimbles.db.db is imported
    an in memory default database is created.
    """ 
    #if not os.path.isdir(path):
    #    raise IOError("No such path '{}'".format(path))
    global _db_url
    _db_url = "sqlite:///{}".format(os.path.abspath(path))

