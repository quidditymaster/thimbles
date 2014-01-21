from widgets import *
from models import *
from views import *

def print_args(*args, **kwargs):
    print args
    print kwargs

class DummyClass(object):
    """just here to give different versions of print args
    """
    def __init__(self, msg):
        self.msg = msg

    def print_args(self, *args, **kwargs):
        print self.msg
        print args
        print kwargs

if __name__ == "__main__":
    qap = QApplication([])
    
    pnw = PrevNext()
    pnw.show()
    
    dc = DummyClass("next happened")
    dc1 = DummyClass("prev happened")
    pnw.next.connect(dc.print_args)
    pnw.prev.connect(dc1.print_args)
    
    qap.exec_()
