import numpy as np
import numpy.lib.recfunctions as np_recfunc
from .utils import int_to_roman, roman_to_int

__all__ = ['periodic_table','solar_abundance','Abundances']

class PeriodicTable (object):
    """
    Store data from the periodic table
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    table_data : Returns a recarray with the information
    species_name : Takes a value or array of species identifiers (see note 1)
        and returns their shortened species name
        
    Raises
    ------
    KeyError : if given species identifier is unknown 

    Notes
    -----
    __1)__ Species identification can be done by proton number or element name (e.g 26 or 'Fe')
        The id is not case sensitive (e.g. 'Fe' == 'fe') and you can but in species
        identification (e.g. ionized iron is 26.1 or 'Fe I' ==> return Fe values)
    
    Examples
    --------    
    ## Accessing Information
    >>> pt[26] # Given the Z
    (26.0, 'Fe', 'Iron', 7.5, 0.0016)
    
    >>> pt[26.1] # It will ignore the ionization
    (26.0, 'Fe', 'Iron', 7.5, 0.0016)
    
    >>> pt['Hg'] # Given by element name (case-insensitive)
    (80.0, 'Hg', 'Mercury', 1.17, 0.0064)
        
    >>> pt['ba II'] # It will ignore ionization if there's a space
    (56.0, 'Ba', 'Barium', 2.18, 0.0081)
    
    >>> # you can also give a list/array of the above ids
    >>> pt[[22.1,22,20.0,25.2,'Fe']] 
    array([(22.0, 'Ti', 'Titanium', 4.95, 0.0025000000000000005),
       (22.0, 'Ti', 'Titanium', 4.95, 0.0025000000000000005),
       (20.0, 'Ca', 'Calcium', 6.34, 0.0016),
       (25.0, 'Mn', 'Maganese', 5.43, 0.0016),
       (26.0, 'Fe', 'Iron', 7.5, 0.0016)], 
      dtype=[('z', '<f8'), ('element', '|S2'), ('element_long', '|S15'), ('abundance', '<f8'), ('sigma', '<f8')])
             
    """    
    
    # this holds the Z,el,element_name
    # converters to get elements, take Z=>el and el=>Z all with arrays
    # TODO : make this a data base to load in
    _table_data = np.array([(1,"H","Hydrogen"),
                     (2,"He","Helium"),
                     (3,"Li","Lithium"),
                     (4,"Be","Beryllium"),
                     (5,"B","Boron"),
                     (6,"C","Carbon"),
                     (7,"N","Nitorgen"),
                     (8,"O","Oxygen"),
                     (9,"F","Florine"),
                     (10,"Ne","Neon"),
                     (11,"Na","Sodium"),
                     (12,"Mg","Magnesium"),
                     (13,"Al","Aluminium"),
                     (14,"Si","Silicon"),
                     (15,"P","Phosphorus"),
                     (16,"S","Slufer"),
                     (17,"Cl","Chlorine"),
                     (18,"Ar","Argon"),
                     (19,"K","Potassium"),
                     (20,"Ca","Calcium"),
                     (21,"Sc","Scandium"),
                     (22,"Ti","Titanium"),
                     (23,"V","Vanadium"),
                     (24,"Cr","Chromium"),
                     (25,"Mn","Maganese"),
                     (26,"Fe","Iron"),
                     (27,"Co","Cobalt"),
                     (28,"Ni","Nickel"),
                     (29,"Cu","copper"),
                     (30,"Zn","Zinc"),
                     (31,"Ga","Gallium"),
                     (32,"Ge","Germanium"),
                     (33,"As","Arsnic"),
                     (34,"Se","Selenium"),
                     (35,"Br","Bormine"),
                     (36,"Kr","Krypton"),
                     (37,"Rb","Rubidium"),
                     (38,"Sr","Strontium"),
                     (39,"Y","Yttrium"),
                     (40,"Zr","Zirconium"),
                     (41,"Nb","Niobium"),
                     (42,"Mo","Molybdenum"),
                     (43,"Tc","Technetium"),
                     (44,"Ru","Ruthenium"),
                     (45,"Rh","Rhodium"),
                     (46,"Pd","Palladium"),
                     (47,"Ag","Silver"),
                     (48,"Cd","Cadmium"),
                     (49,"In","Indium"),
                     (50,"Sn","Tin"),
                     (51,"Sb","Antimony"),
                     (52,"Te","Tellurium"),
                     (53,"I","Iodine"),
                     (54,"Xe","Xenon"),
                     (55,"Cs","Caesium"),
                     (56,"Ba","Barium"),
                     (57,"La","Lanthanum"),
                     (58,"Ce","Cerium"),
                     (59,"Pr","Praseodymium"),
                     (60,"Nd","Neodymium"),
                     (61,"Pm","Promethium"),
                     (62,"Sm","Samarium"),
                     (63,"Eu","Europium"),
                     (64,"Gd","Gadolinium"),
                     (65,"Tb","Terbium"),
                     (66,"Dy","Dyspeosium"),
                     (67,"Ho","Holmium"),
                     (68,"Er","Erbium"),
                     (69,"Tm","Thulium"),
                     (70,"Yb","Ytterbium"),
                     (71,"Lu","Luletium"),
                     (72,"Hf","Hafnium"),
                     (73,"Ta","Tantalum"),
                     (74,"W","Tungsten"),
                     (75,"Re","Rhenium"),
                     (76,"Os","Osmium"),
                     (77,"Ir","iridium"),
                     (78,"Pt","Plantinum"),
                     (79,"Au","Gold"),
                     (80,"Hg","Mercury"),
                     (81,"Tl","Thallium"),
                     (82,"Pb","Lead"),
                     (83,"Bi","Bismuth"),
                     (84,"Po","Polomium"),
                     (85,"At","Astatine"),
                     (86,"Rn","Radon"),
                     (87,"Fr","Francium"),
                     (88,"Ra","Radium"),
                     (89,"Ac","Actinium"),
                     (90,"Th","Thorium"),
                     (91,"Pa","Protactinium"),
                     (92,"U","Uranium"),
                     (93,"Np","Neptunium"),
                     (94,"Pu","Plutonium"),
                     (95,"Am","Americium")],
                    dtype=[('z',float),('element','a2'),('element_long','a15')])
    
    def __init__ (self):
        self._by_z = {}
        self._by_el = {}
        
        for i,row in enumerate(self._table_data):
            self._by_z[row[0]] = i
            self._by_el[row[1].lower()] = i

    @property
    def table_data (self):
        return self._table_data
    
    def _single_species_name (self,spe):
        
        if isinstance(spe,int):
            return self[spe][1]
        
        elif isinstance(spe,float):
            el = self[spe][1]
            ionz = int(round((spe - round(spe-0.5))*10))+1
            return el+" "+int_to_roman(ionz)
        
        else:
            return 'unknown'

    def species_name (self,spe):
        """
        Takes a species id and converts to a name representation
        
        species_id := proton_number + 0.1*(ionization_state)
        where the ionization_state is 0 for ground, 1 for singally, etc
        
        """
        if isinstance(spe,(float,int)):
            return self._single_species_name(spe)
        
        elif isinstance(spe,(list,tuple)):    
            return [self._single_species_name(s) for s in spe]
        elif isinstance(spe,np.ndarray):
            data = self.table_data['element']
            if isinstance(spe.dtype,int):
                return data[self.table_dat['']]
        else:
            raise TypeError("must receive either a floating value of the species or an array")

    def _single_species_id (self,name):
        """
        Takes name, e.g. 'Fe I', and returns the id 26.1
        
        """
        if isinstance(name,basestring):
            sname = name.split()
            if not len(sname):
                return 0.0
            elif len(sname) == 1:
                return self[sname[0]][0]
            elif len(sname) == 2:
                el = self[sname[0]][0]
                ion = sname[1]
                ionz = roman_to_int(ion)
                return el+(ionz-1)*0.1

    def species_id (self,name):
        if isinstance(name,basestring):
            return self._single_species_id(name)
        
        elif isinstance(name,(list,tuple,np.ndarray)):    
            return [self._single_species_id(s) for s in name]
        else:
            raise TypeError("must receive either a string value of the species or an array")

    def __iter__ (self):
        return iter(self.table_data)

    def __contains__ (self,spe):
        if isinstance(spe,str):
            return spe.lower().strip().split()[0] in np.core.defchararray.lower(self.table_data['element'])
        
        if isinstance(spe,(int,float)):
            return int(spe) in self.table_data['z']
    
    def __reversed__ (self):
        return reversed(self.table_data)
    
    def __len__ (self):
        return self.table_data.shape[0]

    def _getitem_single (self,spe):
        if isinstance(spe,str):
            return self.table_data[self._by_el[spe.lower().strip().split()[0]]]
        
        if isinstance(spe,(int,float)):
            return self.table_data[self._by_z[int(spe)]]

        return None

    def _getitem (self,spe):
        """ check input and return appropriate output """
        
        value = self._getitem_single(spe)
        if value is not None:
            return value

        if isinstance(spe,(tuple,list)):
            return np.array([self._getitem_single(s) for s in spe],dtype=self.table_data.dtype)
            
        if isinstance(spe,np.ndarray):
            if spe.ndim != 1:
                raise ValueError("numpy array must be single dimensional")
            
            if spe.shape[0] == 0: 
                return np.array([],dtype=self.table_data.dtype)

            if spe.dtype.type == np.str_:
                return np.array([self._getitem_single(s) for s in spe],dtype=self.table_data.dtype)
                            
            idx = np.round(spe)-1
            return self.table_data[idx.astype(int)]
      
    def __getitem__ (self,spe):
        try: 
            return self._getitem(spe)
        except KeyError:
            raise KeyError("Invalid element name or Z given in : "+str(spe))


# TODO: this stuff
# IDEAS FOR ENHANCEMENT
# class EnhancedPeriodicTable (PeriodicTable):
#     # singleton  --  i.e. global stuff
#     abundance_system = 'lodders' 
#         
#     def to_logeps (self,*args):
#         pass
# 
#     def to_xfe (self, spe, logeps, feh):
#         # spe = [26,28,12]
#         # logeps [7.5,1.3,1.2] stored in Abundance()
#         # Abundance.feh = 7.5-solar_ab
#         # retuns [(xfe,err),...] for spe
#         pass
# 
#     def abundances (self):
#         pass
#         
#         
# class Abundance ():
#     
#     periodic_table = EnhancedPeriodicTable
#     
#     
#     def __init__ (self):
#         self.abundances = []
#         self.abundance_format = 'xfe'
#     
#     def get(self, species, scale="logeps"):
#         pass
#     
#     def to_xfe (self,species):
#         # take self.abundances return xfe
#         pass
#     
        

periodic_table = PeriodicTable()

class Abundances (PeriodicTable):
    """ 
    Stores the data for a particular system of abundances to normalize to
    
    also see help(PeriodicTable)
    
    Parameters
    ----------
    citation : string
        The citation information for where the abundance standard was taken
    data : array like
        Has data [[z,abund,stdev],[...],...] you can optionally include stdev
    normalization : 'H','Si'
        This is the scale the data is on, 'H' ==> H=10**12, 'Si'==> Si=10**6
    
    Attributes
    ----------
    citation : string
        Returns a string which has the input citation information
    array : np.ndarray
        Returns an floatting point array of [[z,abund,sigma],[...],...]
    to_logeps : function
        Takes values from bracket notation and converts to logeps
    to_xfe : function
        Takes values from logeps and converts to bracket notation
    solar_abundance : function
        Takes a value or array of element ids (Z,element_name) and returns
        the solar abundance for those values
    current_normalization : Refers to what scale the abundances were normalized
        to. Can change the normalization by setting this attribute
        
    Raises
    ------
    KeyError : if given element identication is unknown 

    Notes
    -----
    __1)__ Element identification can be done by proton number or element name (e.g 26 or 'Fe')
        The id is not case sensitive (e.g. 'Fe' == 'fe') and you can but in species
        identification (e.g. ionized iron is 26.1 or 'Fe I' ==> return Fe values)
        
    """
    # For every value on the periodic_table it has a corresponding (abund,error)
    # have ways to input Z,el as value or array and get back abund,error
    
    # methods for [x/fe],[x/y] and logeps conversions
    def __init__ (self,data=None,normalization='H'):        
        super(Abundances,self).__init__()
        
        # get all the data and map it to the values in the periodic table
        by_z = {}
        if data is None:
            data = []
        else:
            if not isinstance(data,(tuple,list,np.ndarray)):
                raise TypeError("data must be iterable object with [[z,abund,uncertainty],[...],...]")
        for row in data:
            try: spe = self[row[0]]
            except KeyError:
                raise ValueError("Reveived unknown element name or Z : "+str(row[0]))
            
            abund = float(row[1])
            if len(row)>2:
                error = float(row[2] or np.inf)**2
            else:
                error = np.inf
            
            by_z[int(spe[0]-1)] = (abund,error)
        
        # create a numpy recarray of the values, filling in missing values  
        abunds = []
        sigma = []
        for i in xrange(len(self)):
            a,v = by_z.get(i,(np.nan,np.inf))
            abunds.append(a)
            sigma.append(v)
                
        # combine the tables
        names = ('abundance','sigma')
        dtypes = (float,float)
        self._table_data = np_recfunc.append_fields(self.table_data,names,(abunds,sigma),dtypes=dtypes,usemask=False,asrecarray=True)
        
        self._normalization_options = ('H','Si')
        self._current_normalization = self._normalization_options[0]
        self.current_normalization = normalization
        
        self._init_str_representation()
    
    def _init_str_representation (self):
        title_str = "  {0:>4} {1:>9} {2:>16} {3:>10} {4:>10}"
        data_str =  " [{0:>4},{1:>9},{2:>16},{3:>9.3f},{4:>9.3f}]"
        
        rep = title_str.format(*tuple(self.table_data.dtype.names))+"\n"
        rowlength = len(rep)
        rep += "-"*rowlength+"\n"
        rep += "["
        
        for i in xrange(0,len(self)-1): 
            row = self.table_data[i]
            rep += data_str.format(*row)+",\n"
        
        rep += data_str.format(*self.table_data[-1])+"]\n"
        rep += "-"*rowlength+"\n"+title_str.format(*tuple(self.table_data.dtype.names))+"\n"
        
        self._str_rep = rep
        
    def __repr__ (self):
        return self.__str__()
        
    def __str__ (self):
        return self._str_rep
    
    @property
    def current_normalization (self):
        return self._current_normalization
    
    @current_normalization.setter
    def current_normalization (self,normalization):
        if not isinstance(normalization,str):
            raise TypeError("Normalization must be a string")
        
        if not normalization:
            raise ValueError("Empty string received")
        
        normalization = normalization[0].upper()+normalization[1:]
        
        if normalization not in self._normalization_options:
            raise ValueError("Normalization must be "+", ".join(self._normalization_options))
            
        # if the value is equal to the current then no change
        if normalization == self._current_normalization:
            return

        (abund_h,_err),(abund_si,_err) = self.solar_abundance((1,14))
        # if in H scale and converting to Si
        if self._current_normalization == "H" and normalization == 'Si':
            if abund_si == 6.0:
                return
            diff = 6.0 - abund_si
            
        # convert Si to H
        elif self._current_normalization == 'Si' and normalization == 'H':
            if abund_h == 12.0:
                return
            diff = 12.0 - abund_h
        else:
            raise StandardError("Whoops, this shouldn't happen")
        
        # TODO: fix the errors
        self._table_data['abundance'] += diff  
        self._init_str_representation()       
        self._current_normalization = normalization
    
    @property
    def array (self):
        return np.dstack((self.table_data['z'].astype(float),self.table_data['abundance'],self.table_data['sigma']))[0]

class AbundanceSystem (Abundances):
    """ 
    Stores the data for a particular system of abundances to normalize to
    
    also see help(PeriodicTable)
    
    Parameters
    ----------
    citation : string
        The citation information for where the abundance standard was taken
    data : array like
        Has data [[z,abund,stdev],[...],...] you can optionally include stdev
    normalization : 'H','Si'
        This is the scale the data is on, 'H' ==> H=10**12, 'Si'==> Si=10**6
    
    Attributes
    ----------
    citation : string
        Returns a string which has the input citation information
    array : np.ndarray
        Returns an floatting point array of [[z,abund,sigma],[...],...]
    to_logeps : function
        Takes values from bracket notation and converts to logeps
    to_xfe : function
        Takes values from logeps and converts to bracket notation
    solar_abundance : function
        Takes a value or array of element ids (Z,element_name) and returns
        the solar abundance for those values
    current_normalization : Refers to what scale the abundances were normalized
        to. Can change the normalization by setting this attribute
        
    Raises
    ------
    KeyError : if given element identication is unknown 

    Notes
    -----
    __1)__ Element identification can be done by proton number or element name (e.g 26 or 'Fe')
        The id is not case sensitive (e.g. 'Fe' == 'fe') and you can but in species
        identification (e.g. ionized iron is 26.1 or 'Fe I' ==> return Fe values)
        
    """
    def __init__ (self,citation,data,normalization='H'):
        if not isinstance(citation,str):
            raise TypeError("Citation must a string for the abundance system standard")        
        self._citation = citation
        super(AbundanceSystem,self).__init__(data,normalization=normalization)
        
    def __repr__ (self):
        return "Abundance System : "+str(self.citation)

    @property
    def citation (self):
        return self._citation
    
    pass
#     def to_xy (self,speX,speY,logepsX,logepsY,varX=None,varY=None):
#         """
#         Convert value from logepsX and logepsY to [X/Y] 
#         
#         [X/Y] = (logeps(X)/logeps(Y)) - (logeps(X)_sun/logeps(Y)_sun)
# 
#         ----------
#         speX,speY : float or array of floats
#             Gives the specie(s) which correspond to the logepsX, logepsY values
# 
#         logepsX,logepsY : float or array of floats
#             Gives the logeps notation of the abundance for the 
#             corresponding species
#             
#         Returns
#         -------
#         if spe is a float:
#             xy : float
#                 The abundance of the species with abundance logeps
#             var : float
#                 The sigma on that abundance
#         
#         if spe is an array of floats:
#             xy : array
#                 The abudnaces for the species with abundances logeps
#         
#             var : array
#                 The sigma on that abundance
#         """
#         logepsX_sun,_varX_sun = self.solar_abundance(speX).T
#         logepsY_sun,_varY_sun = self.solar_abundance(speY).T
#         return (logepsX/logepsY)-(logepsX_sun/logepsY_sun), np.nan
    
    def to_logeps (self, spe, xfe, feh):
        """
        
        Convert value in [X/Fe] to logeps solar hydrogen scale
        
        [Fe/H] = logeps(Fe) - logeps(Fe)_sun
        logeps(X) = [X/Fe] + ([Fe/H] + logeps(X)_sun)
    
        Parameters
        ----------
        spe : float or array of floats
            Gives the specie(s) which correspond to the [X/Fe] values
        xfe : float or array of floats
            Gives the [X/Fe] bracket notation of the abundance for the 
            corresponding species
        feh : float
            the [Fe/H] value to be used for the converstion
            
        Returns
        -------
        if spe is a float:
            logeps : float
                The abundance of the species with abundance [X/Fe]
        if spe is an array of floats:
            logeps : array
                The abudnaces for the species with abundances [X/Fe]

        """
        # TODO: if xfe is given by shape(N,2) then assume there are errors
        # and calculate them correctly
        return xfe + (feh + self.solar_abundance(spe)[:,0])
        
    def to_xfe (self, spe, logeps, feh):
        """
        
        Convert value in logeps solar hydrogen scale to [X/Fe]
        
        [Fe/H] = logeps(Fe) - logeps(Fe)_sun
        [X/Fe] = logeps(X) - ([Fe/H] + logeps(X)_sun)

        Parameters
        ----------
        spe : float or array of floats
            Gives the specie(s) which correspond to the logeps values
        xfe : float or array of floats
            Gives the logeps notation of the abundance for the 
            corresponding species
        feh : float
            the [Fe/H] value to be used for the converstion
            
        Returns
        -------
        if spe is a float:
            xfe : float
                The abundance of the species with abundance logeps
        if spe is an array of floats:
            xfe : array
                The abudnaces for the species with abundances logeps
                
        """
        return logeps - (feh + self.solar_abundance(spe)[:,0])

    def to_numberdensity (self,logeps):
        return np.power(10.0, self.solar_abundance(self.current_normalization)-logeps)

    def solar_abundance (self,spe):
        """
        Get the solar logeps abundance for a species (or list of species)
        
        Parameters
        ----------
        spe : float or array of floats
            Give the species identification for which to return the solar abundance
                    
        Returns
        -------
        if spe is float:
            logeps : float
        if spe is array of floats:
            logeps : array
        """
        
        getitem = self.__getitem__(spe)
        
        if getitem is None:
            return None
        
        if isinstance(getitem,np.void):
            return np.array([tuple(getitem)[3:5]])
        else:
            return np.dstack((getitem['abundance'],getitem['sigma']))[0] 
    
class EditableAbundances (Abundances):
    
    # TODO: create this to hold abundances. 
    def __init__ (self,data=None,normalization='H'):
        super(EditableAbundances,self).__init__(data,normalization=normalization)
        self._changed = False
        
    def set_abundances (self,spe,abundances,errors):
        self._changed = True
            
    def __str__ (self):
        if self._changed:
            self._init_str_representation()
        return self._str_rep
        
class AbundanceSystemsDict (dict):

    def __doc__ (self):
        """
        Special dictionary which only holds AbundanceSystem object
        but otherwise acts like a normal dictionary
        """
    
    def _check_abundsys_in (self,abundance_system):
        if not isinstance(abundance_system,AbundanceSystem):
            raise TypeError("received wrong type of abundance_system")

    def __setitem__ (self,name,abundance_system):
        self._check_abundsys_in(abundance_system)
        if name in self:
            return
        super(AbundanceSystemsDict,self).__setitem__(name,abundance_system)
           
class SolarAbundance (AbundanceSystem):
    """ 
    Stores the data for a particular system of abundances to normalize to
    
    also see help(PeriodicTable)
    
    Parameters
    ----------
    abundance_systems : AbundanceSystemsDict
        This holds all the abundance systems you can switch between
    system : string
        Gives which should be used initially
        
    Attributes
    ----------
    current_system : The abundance system which is being used. Who's measurements
        of solar abundances. Can change the current_system by setting this attribute
            
    ## Based on the current system:
    
    citation : string
        Returns a string which has the input citation information
    array : np.ndarray
        Returns an floatting point array of [[z,abund,sigma],[...],...]
    to_logeps : function
        Takes values from bracket notation and converts to logeps
    to_xfe : function
        Takes values from logeps and converts to bracket notation
    solar_abundance : Takes a value or array of element ids (Z,element_name) and returns
        the solar abundance for those values
    current_normalization : Refers to what scale the abundances were normalized
        to. Can change the normalization by setting this attribute
    
    Raises
    ------
    KeyError : if given element identication is unknown 

    Notes
    -----
    __1)__ Element identification can be done by proton number or element name (e.g 26 or 'Fe')
        The id is not case sensitive (e.g. 'Fe' == 'fe') and you can but in species
        identification (e.g. ionized iron is 26.1 or 'Fe I' ==> return Fe values)

    Examples
    --------
    >>> ab = SolarAbundance(abundance_systems)
    
    ## Accessing Information
    
    >>> ab[26] # Given the Z
    (26.0, 'Fe', 'Iron', 7.5, 0.0016)
    
    >>> ab[26.1] # It will ignore the ionization
    (26.0, 'Fe', 'Iron', 7.5, 0.0016)
    
    >>> ab['Hg'] # Given by element name (case-insensitive)
    (80.0, 'Hg', 'Mercury', 1.17, 0.0064)
        
    >>> ab['ba II'] # It will ignore ionization if there's a space
    (56.0, 'Ba', 'Barium', 2.18, 0.0081)
    
    >>> # you can also give a list/array of the above ids
    >>> ab[[22.1,22,20.0,25.2,'Fe']] 
    array([(22.0, 'Ti', 'Titanium', 4.95, 0.0025000000000000005),
       (22.0, 'Ti', 'Titanium', 4.95, 0.0025000000000000005),
       (20.0, 'Ca', 'Calcium', 6.34, 0.0016),
       (25.0, 'Mn', 'Maganese', 5.43, 0.0016),
       (26.0, 'Fe', 'Iron', 7.5, 0.0016)], 
      dtype=[('z', '<f8'), ('element', '|S2'), ('element_long', '|S15'), ('abundance', '<f8'), ('sigma', '<f8')])
    
    ## Built in Methods
    
    >>> ab.to_xfe(22, 3.1, -2.3)
    array([ 0.45])
    
    >>> ab.to_xfe([22,22,20,25],[3.1,3.2,4.4,3.1],-2.3)
    array([ 0.45,  0.55,  0.36, -0.03])
    
    ## Can Change the Normalization
    >>> ab.current_normalization, ab['Si']
    'H', (14.0, 'Si', 'Silicon', 7.51, 0.0009)
    
    >>> ab.current_normalizaion = 'Si'
    
    >>> ab.current_normalization, ab['Si']
    'Si', (14.0, 'Si', 'Silicon', 6.0, 0.0009)
    
    ## Can Change which system is being used
    
    >>> ab.current_system, ab.systems
    'a09', ['a09', 'l03', 'moog']
    
    >>> ab.current_system = 'moog' # for data in Batom.f
    

        
    """             
    def __init__ (self,abundance_systems,system=None):
        super(SolarAbundance,self).__init__('None',[])
        
        if not isinstance(abundance_systems,AbundanceSystemsDict):
            raise TypeError("Abundance system must be of class AbundanceSystemsDict")
        
        if len(abundance_systems) == 0:
            raise ValueError("must initiate with none empty abundance_systems")
        
        # internal dictionary of all abundance systems
        self._abundance_systems = abundance_systems
        
        # set which system to use
        
        if system is None:
            self.current_system = abundance_systems.keys()[-1]
        else:
            self.current_system = system
     
    def __doc__ (self):
        super(SolarAbundance,self).__doc__()
     
    @property
    def current_system (self):
        return self._current_system
    
    @current_system.setter
    def current_system (self,system_):        
        if system_ not in self._abundance_systems:
            raise ValueError("Unknown abundance system : "+str(system_))
        # check value is appropriate
        abundance_system = self._abundance_systems[system_]
        
        self._current_system = system_
        self._table_data = abundance_system._table_data
        self._citation = abundance_system._citation
        self._current_normalization = abundance_system._current_normalization
        self._str_rep = abundance_system._str_rep
                 
    @property
    def systems (self):
        return self._abundance_systems.keys()

def _data_sets ():
    # TODO: Write doc string
    #  Changed the data to be stored
    _batom = [(1, 12.0, np.inf),
              (2, 10.99, np.inf),
              (3, 3.31, np.inf),
              (4, 1.42, np.inf),
              (5, 2.88, np.inf),
              (6, 8.56, np.inf),
              (7, 8.05, np.inf),
              (8, 8.93, np.inf),
              (9, 4.56, np.inf),
              (10, 8.09, np.inf),
              (11, 6.33, np.inf),
              (12, 7.58, np.inf),
              (13, 6.47, np.inf),
              (14, 7.55, np.inf),
              (15, 5.45, np.inf),
              (16, 7.21, np.inf),
              (17, 5.5, np.inf),
              (18, 6.56, np.inf),
              (19, 5.12, np.inf),
              (20, 6.36, np.inf),
              (21, 3.1, np.inf),
              (22, 4.99, np.inf),
              (23, 4.0, np.inf),
              (24, 5.67, np.inf),
              (25, 5.39, np.inf),
              (26, 7.52, np.inf),
              (27, 4.92, np.inf),
              (28, 6.25, np.inf),
              (29, 4.21, np.inf),
              (30, 4.6, np.inf),
              (31, 2.88, np.inf),
              (32, 3.41, np.inf),
              (33, 2.37, np.inf),
              (34, 3.35, np.inf),
              (35, 2.63, np.inf),
              (36, 2.23, np.inf),
              (37, 2.6, np.inf),
              (38, 2.9, np.inf),
              (39, 2.24, np.inf),
              (40, 2.6, np.inf),
              (41, 1.42, np.inf),
              (42, 1.92, np.inf),
              (43, 0.0, np.inf),
              (44, 1.84, np.inf),
              (45, 1.12, np.inf),
              (46, 1.69, np.inf),
              (47, 1.24, np.inf),
              (48, 1.86, np.inf),
              (49, 0.82, np.inf),
              (50, 2.0, np.inf),
              (51, 1.04, np.inf),
              (52, 2.24, np.inf),
              (53, 1.51, np.inf),
              (54, 2.23, np.inf),
              (55, 1.12, np.inf),
              (56, 2.13, np.inf),
              (57, 1.22, np.inf),
              (58, 1.55, np.inf),
              (59, 0.71, np.inf),
              (60, 1.5, np.inf),
              (61, 0.0, np.inf),
              (62, 1.0, np.inf),
              (63, 0.51, np.inf),
              (64, 1.12, np.inf),
              (65, 0.33, np.inf),
              (66, 1.1, np.inf),
              (67, 0.5, np.inf),
              (68, 0.93, np.inf),
              (69, 0.13, np.inf),
              (70, 1.08, np.inf),
              (71, 0.12, np.inf),
              (72, 0.88, np.inf),
              (73, 0.13, np.inf),
              (74, 0.68, np.inf),
              (75, 0.27, np.inf),
              (76, 1.45, np.inf),
              (77, 1.35, np.inf),
              (78, 1.8, np.inf),
              (79, 0.83, np.inf),
              (80, 1.09, np.inf),
              (81, 0.82, np.inf),
              (82, 1.85, np.inf),
              (83, 0.71, np.inf),
              (84, 0.0, np.inf),
              (85, 0.0, np.inf),
              (86, 0.0, np.inf),
              (87, 0.0, np.inf),
              (88, 0.0, np.inf),
              (89, 0.0, np.inf),
              (90, 0.12, np.inf),
              (91, 0.0, np.inf),
              (92, 0.0, np.inf),
              (93, 0.0, np.inf),
              (94, 0.0, np.inf),
              (95, 0.0, np.inf)]
    
    
    _lodders03 =[("h", 12.0, 0.0),
                ("he", 10.899, 0.01),
                ("li", 3.28, 0.06),
                ("be", 1.41, 0.08),
                ("b", 2.78, 0.04),
                ("c", 8.39, 0.04),
                ("n", 7.83, 0.11),
                ("o", 8.69, 0.05),
                ("f", 4.46, 0.06),
                ("ne", 7.87, 0.1),
                ("na", 6.3, 0.03),
                ("mg", 7.55, 0.02),
                ("al", 6.46, 0.02),
                ("si", 7.54, 0.02),
                ("p ", 5.46, 0.04),
                ("s ", 7.19, 0.04),
                ("cl", 5.26, 0.06),
                ("ar", 6.55, 0.08),
                ("k ", 5.11, 0.05),
                ("ca", 6.34, 0.03),
                ("sc", 3.07, 0.04),
                ("ti", 4.92, 0.03),
                ("v ", 4.0, 0.03),
                ("cr", 5.65, 0.05),
                ("mn", 5.5, 0.03),
                ("fe", 7.47, 0.03),
                ("co", 4.91, 0.03),
                ("ni", 6.22, 0.03),
                ("cu", 4.26, 0.06),
                ("zn", 4.63, 0.04),
                ("ga", 3.1, 0.06),
                ("ge", 3.62, 0.05),
                ("as", 2.32, 0.05),
                ("se", 3.36, 0.04),
                ("br", 2.59, 0.09),
                ("kr", 3.28, 0.08),
                ("rb", 2.36, 0.06),
                ("sr", 2.91, 0.04),
                ("y ", 2.2, 0.03),
                ("zr", 2.6, 0.03),
                ("nb", 1.42, 0.03),
                ("mo", 1.96, 0.04),
                ("ru", 1.82, 0.08),
                ("rh", 1.11, 0.03),
                ("pb", 1.7, 0.03),
                ("ag", 1.23, 0.06),
                ("cd", 1.74, 0.03),
                ("in", 0.8, 0.03),
                ("sn", 2.11, 0.04),
                ("sb", 1.06, 0.07),
                ("te", 2.22, 0.04),
                ("i", 1.54, 0.12),
                ("xe", 2.27, 0.02),
                ("cs", 1.1, 0.03),
                ("ba", 2.18, 0.03),
                ("la", 1.18, 0.06),
                ("ce", 1.61, 0.02),
                ("pr", 0.78, 0.03),
                ("nd", 1.46, 0.03),
                ("sm", 0.95, 0.04),
                ("eu", 0.52, 0.04),
                ("gd", 1.06, 0.02),
                ("tb", 0.31, 0.03),
                ("dy", 1.13, 0.04),
                ("ho", 0.49, 0.02),
                ("er", 0.95, 0.03),
                ("tm", 0.11, 0.06),
                ("yb", 0.94, 0.03),
                ("lu", 0.09, 0.06),
                ("hf", 0.77, 0.04),
                ("ta", -0.14, 0.03),
                ("w", 0.65, 0.03),
                ("re", 0.26, 0.04),
                ("os", 1.37, 0.03),
                ("ir", 1.35, 0.03),
                ("pt", 1.67, 0.03),
                ("au", 0.83, 0.06),
                ("hg", 1.16, 0.18),
                ("tl", 0.81, 0.04),
                ("pb", 2.05, 0.04),
                ("bi", 0.68, 0.03),
                ("th", 0.09, 0.04),
                ("u", -0.49, 0.04)]
    
    # the forth column is a flag
    # 0 = good photosphere measurement
    # 1 = uncertain photosphere measurement
    # 2 = good meteoritic measurement
    # Hydrogen has no error or flag
    _asplund09 =  [("H" , 12.0, np.inf, np.nan),
                  ("He" , 10.93, 0.01, 1),
                  ("Li" , 1.05, 0.1, 0),
                  ("Be" , 1.38, 0.09, 0),
                  ("B" , 2.7, 0.2, 0),
                  ("C" , 8.43, 0.05, 0),
                  ("N" , 7.83, 0.05, 0),
                  ("O" , 8.69, 0.05, 0),
                  ("f" , 4.56, 0.3, 0),
                  ("ne" , 7.93, 0.1, 1),
                  ("Na" , 6.24, 0.04, 0),
                  ("mg" , 7.6, 0.04, 0),
                  ("al" , 6.45, 0.03, 0),
                  ("si" , 7.51, 0.03, 0),
                  ("p" , 5.41, 0.03, 0),
                  ("s" , 7.12, 0.03, 0),
                  ("cl" , 5.5, 0.3, 0),
                  ("ar" , 6.4, 0.13, 1),
                  ("k" , 5.03, 0.09, 0),
                  ("ca" , 6.34, 0.04, 0),
                  ("sc" , 3.15, 0.04, 0),
                  ("ti" , 4.95, 0.05, 0),
                  ("v" , 3.93, 0.08, 0),
                  ("cr" , 5.64, 0.04, 0),
                  ("mn" , 5.43, 0.04, 0),
                  ("fe" , 7.5, 0.04, 0),
                  ("co" , 4.99, 0.07, 0),
                  ("ni" , 6.22, 0.04, 0),
                  ("cu" , 4.19, 0.04, 0),
                  ("zn" , 4.56, 0.05, 0),
                  ("ga" , 3.04, 0.09, 0),
                  ("ge" , 3.65, 0.1, 0),
                  ("as" , 2.30, 0.04, 2),
                  ("se" , 3.34, 0.03, 2),
                  ("br" , 2.54, 0.06, 2), 
                  ("kr" , 3.25, 0.06, 1),
                  ("rb" , 2.52, 0.1, 0),
                  ("sr" , 2.87, 0.07, 0),
                  ("y" , 2.21, 0.05, 0),
                  ("zr" , 2.58, 0.04, 0),
                  ("nb" , 1.46, 0.04, 0),
                  ("mo" , 1.88, 0.08, 0),
                  ("ru" , 1.75, 0.08, 0),
                  ("rh" , 0.91, 0.1, 0),
                  ("pd" , 1.57, 0.1, 0),
                  ("ag" , 0.94, 0.1, 0),
                  ("cd" , 1.71, 0.03, 2),
                  ("in" , 0.8, 0.2, 0),
                  ("sn" , 2.04, 0.1, 0),
                  ("sb" , 1.01, 0.06, 2),
                  ("te" , 2.18, 0.03, 2),
                  ("i" , 1.55, 0.08, 2),
                  ("xe" , 2.24, 0.06, 1),
                  ("cs" , 1.08, 0.02, 2),
                  ("ba" , 2.18, 0.09, 0),
                  ("la" , 1.1, 0.04, 0),
                  ("ce" , 1.58, 0.04, 0),
                  ("pr" , 0.72, 0.04, 0),
                  ("nd" , 1.42, 0.04, 0),
                  ("sm" , 0.96, 0.04, 0),
                  ("eu" , 0.52, 0.04, 0),
                  ("gd" , 1.07, 0.04, 0),
                  ("tb" , 0.3, 0.1, 0),
                  ("dy" , 1.1, 0.04, 0),
                  ("ho" , 0.48, 0.11, 0),
                  ("er" , 0.92, 0.05, 0),
                  ("tm" , 0.1, 0.04, 0),
                  ("yb" , 0.84, 0.11, 0),
                  ("lu" , 0.1, 0.09, 0),
                  ("hf" , 0.85, 0.04, 0),
                  ("ta" , -0.12, 0.04, 2),
                  ("w" , 0.85, 0.12, 0),
                  ("re" , 0.26, 0.04, 2),
                  ("os" , 1.4, 0.08, 0),
                  ("ir" , 1.38, 0.07, 0),
                  ("pt" , 1.26, 0.03, 2),
                  ("au" , 0.92, 0.1, 0),
                  ("hg" , 1.17, 0.08, 2),
                  ("tl" , 0.9, 0.2, 0),
                  ("pb" , 1.75, 0.1, 0),
                  ("bi" , 0.65, 0.04, 2),
                  ("th" , 0.02, 0.1, 0),
                  ("u" , -0.54, 0.03, 0)]
    
    return _batom,_lodders03,_asplund09

_batom,_lodders03,_asplund09 = _data_sets()

# declare each of the abundance systems here
l03 = AbundanceSystem(('Lodders, K. 2003 Solar System Abundances and Condensation ' \
                      'Temperatures of the Element, APJ, 591:1220-1247'),_lodders03)

asplund09 = AbundanceSystem(("Asplund, M., Grevesse, N., Sauval, A. Jacques, Scott, P. "\
                             "2009 The Chemical Composition of the Sun. Annual Review of "\
                             "Astronomy and Astrophysics. 47:481-522 "),_asplund09)

moog = AbundanceSystem(("Batom.f 'the set of current solar (when available) or meteorite "       
                         'abundances, scaled to log(h) = 12.00 .  The data are from Anders '
                         'and Grevesse (1989, Geochim.Cosmichim.Acta, v53, p197) and the solar '
                         "values are adopted except for a) uncertain solar data, or b) Li, Be, "
                         "and B, for which the meteoritic values are adopted. " 
                         "I was told to use the new Fe value of 7.52 as adopted in Sneden "
                         "et al. 1992 AJ 102 2001.'"),_batom)

# combine all solar abundance systems into one dictionary
abundance_systems = AbundanceSystemsDict({'l03':l03,'moog':moog,'a09':asplund09})

# create the useful solar_abundance object using all the possible systems
solar_abundance = SolarAbundance(abundance_systems,system='a09')
