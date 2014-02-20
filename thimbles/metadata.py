# Standard Library
from __future__ import print_function, division
from copy import deepcopy

# 3rd Party

# Internal

# ########################################################################### #

class MetaData (dict):
    """
    A class for holding information about an object
    """
    
    def __init__ (self,*args,**kwargs):
        super(MetaData,self).__init__(*args,**kwargs)
    
    def __repr__ (self):
        reprout = 'MetaData {'
        if len(self) == 0:
            return reprout + "}"
        reprout += "\n"
        for key in self:
            value = str(repr(self[key])).split("\n")
            reprout += " "+str(key)+" : "
            reprout += value[0].strip()+"\n"
            if len(value) > 1: 
                reprout += " "*(len(key))+"    ...\n"
        reprout += "}\n"
        return reprout
    
    def __str__ (self):
        return super(MetaData,self).__repr__()
    
    def _type_check_other (self,other):
        if not isinstance(other,dict):
            raise TypeError("other must be a subclass of dict")
    
    def __add__ (self,other):
        return self.combine(other,key_conflicts='raise')
            
    def __iadd__ (self,other):
        self._type_check_other(other)
        for key in other:
            if self.has_key(key):
                continue
            self[key] = other[key]
        return self    
        
    def combine (self,other,key_conflicts='ignore',return_=False):
        """
        Combine two MetaData dictionaries together. 
        
        
        Parameters
        ----------
        other : dict subclass
            Any dictionary object will work including other MetaData Dictionaries
        key_conflicts : 'ignore' (default), 'merge', 'warn', 'raise'
            Defined the method to handle key conflicts
            * ignore : if key is in conflict, keep the current key with no warning
            * merge : convert key to string and add integers until unique key is found
            * warn : print a warning message for key conflicts. Keep current key
            * raise : raise error message for key conflicts.
        return_ : boolean
            If True then it will keep the data in place and return a copy with
            with the concatenation
                    
        Returns
        -------
        info : MetaData 
            Returns an information object with keys and information 
            concatenated from the two
        
        
        Raises
        ------
        KeyError : If key_conflicts=='raise' is True and conflicts exist between two keys
        
        
        Notes
        -----
        __1)__ If a key is in conflict but the data the key refers to is the same then
            no messages or errors will be raised
        
        
        Special cases
        -------------
        add operator : info1 + info2
            This will raise errors for key conflicts between the two 
        iadd operator : info1 += info2
            This will ignore key conflicts 
            and always takes info1 keys as default
            
        """
        self._type_check_other(other)
        def errmsg (key):
            return "Warning: key conflict '"+str(key)+"'"
        
        key_conflicts = key_conflicts.lower()
        if return_:
            out = self.copy()
        else:
            out = self
        
        if key_conflicts=='merge':
            for key in other:
                if self.has_key(key) and self[key]==other[key]:
                    continue
                i = 0   
                base_key = deepcopy(key)
                while self.has_key(key):            
                    key = str(base_key)+"_"+str(i)
                    i += 1    
                out[key] = other[base_key]
            return out
        # else:
        for key in other:
            if self.has_key(key):
                # if the data's the same don't worry about it
                if self[key]==other[key]:
                    continue
                # resolve conflicts 
                if key_conflicts=='raise':
                    raise KeyError(errmsg(key))
                elif key_conflicts=='warn':
                    print(errmsg(key))
                else:
                    continue
            out[key] = other[key]
        
        if return_:
            return out

    def copy (self):
        return deepcopy(self)
    
    def header_list(self):
        """returns a list of the values belonging to keys beginning header_ """
        keys = self.keys()
        headers = []
        for key in keys:
            try:
                keystart = key[:7]
                if keystart == "header_":
                    headers.append(self[key])
            except:
                pass
        return headers
            
    def guess_observation_time(self, headers=None):
        if headers == None:
            headers = self.header_list()
        obs_time = None
        for hdr in headers:
            try:
                obs_time = hdr["ut"]
                break
            except:
                pass
        return obs_time
    
    def guess_airmass(self, headers):
        if headers == None:
            headers = self.header_list()
        airmass = None
        for hdr in headers:
            try:
                airmass = hdr["airmass"]
                break
            except:
                pass
        return airmass
    
    def guess_object_name(self):
        return None

