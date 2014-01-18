import pdb #@UnusedImport


def int_to_roman(value):
    """
    Convert an integer to Roman numerals.
    
    Reference: http://code.activestate.com/recipes/81611-roman-numerals/
    
    Parameters
    ----------
    value : integer
        The integer used to convert to Roman
    
    Returns
    -------
    roman : string
        A string with the corresponding Roman of the value integer
    
    Raises
    ------
    TypeError : If value is not an integer
    ValueError : If the value is not in 0 < value < 4000
    
         
    Examples
    --------
    >>> int_to_roman(0)
    Traceback (most recent call last):
    ValueError: Argument must be between 1 and 3999
    
    >>> int_to_roman(-1)
    Traceback (most recent call last):
    ValueError: Argument must be between 1 and 3999
    
    >>> int_to_roman(1.5)
    Traceback (most recent call last):
    TypeError: expected integer, got <type 'float'>
    
    >>> print int_to_roman(2000)
    MM
    >>> print int_to_roman(1999)
    MCMXCIX
    
    """
    if not isinstance(value,int):
        raise TypeError("expected integer, got %s" % type(value))
    if not 0 < value < 4000:
        raise ValueError("Argument must be between 1 and 3999")   
    ints = (1000,  900, 500, 400, 100,  90,  50,  40, 10,   9,  5,   4,  1)
    nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
    result = ""
    for i in range(len(ints)):
        count   = int(value/ints[i])
        result += nums[i]*count
        value  -= ints[i]*count
    return result

class SpreadsheetCells (dict):
    """
    This is a subclass of a dictionary which has extra restrictions for the
    keys and values. Keys must be (row,column) and not overwrite unless 
    explicitly so. Values must be (value,style) for later putting into
    an xlwt spreadsheet
    
    """
    
    def __init__ (self):
        super(SpreadsheetCells,self).__init__()
    
    def _check_in (self,i):
        try: _row,_col = i 
        except TypeError:
            raise TypeError("Input into ablines must be (row,col) ")
            
    def _check_set (self,y):
        try: _input,_style = y 
        except TypeError:
            raise TypeError("Set value must be (input,style)") 
           
    def __setitem__ (self,i,y):
        self._check_in(i)
        self._check_set(y)
        if i in self:
            pdb.set_trace()
            raise KeyError("Key already given"+str(i))

        super(SpreadsheetCells,self).__setitem__(i,y)
    
    def overwrite (self,i,y):
        self._check_in(i)
        self._check_set(y)
        super(SpreadsheetCells,self).__setitem__(i,y)
        
    def get (self,i,y):
        self._check_in(i)
        self._check_set(y)
        return super(SpreadsheetCells,self).get(i,y)

    def __delitem__ (self,i):
        self._check_in(i)
        super(SpreadsheetCells,self).__delitem__(i)
