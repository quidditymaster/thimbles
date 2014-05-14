

# ########################################################################### #
from __future__ import print_function

class Verbosity (object): 
    
    
    def __init__ (self,level='silent'):
        """
        Parameters
        ----------
        level : 'silent', 
        
        
        """
        self.levels = {'silent':50,
                       'verbose':10}        
        self.set_level(level)
    
    def set_level (self,level):
        self.current_level = int(self.levels.get(level,level))
            
    def __call__ (self,msg,level='verbose'): 
        lvl = int(self.levels.get(level,level))
        if lvl >= self.current_level and self.current_level != self.levels['silent']:
            print(msg) 