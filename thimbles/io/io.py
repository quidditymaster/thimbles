# Purpose: For reading in data from fits and text files
# Authors: Dylan Gregersen, Tim Anderton
# Date: 2013/08/13 23:52:42
#

# ########################################################################### #
# Standard Library
import re
import os
import time
from copy import deepcopy
from collections import OrderedDict, Iterable

# 3rd Party
import scipy
from astropy.io import fits
import cPickle as pickle
import numpy as np

# Internal
from ..utils.misc import smoothed_mad_error
from ..utils.misc import var_2_inv_var
from ..spectrum import Spectrum, WavelengthSolution

# ########################################################################### #

__all__ = ["read","read_txt","read_fits",
           "query_fits_header","WavelengthSolutionFunctions",
           "ExtractWavelengthCoefficients"
           ]

# ########################################################################### #

# NOTE: Could also check for 'END' card and/or 'NAXIS  ='
_fits_re = re.compile("[SIMPLE  =,XTENSION=]"+"."*71+"BITPIX  =")

pass
# ############################################################################ #
class query_fits_header:
    """ 
    This class looks through a fits header object and formats queries into useable formats
    
    EXAMPLE:
    >>> cval = __gethdr__(fits_header,'CVAL',no_val=0)
    >>> # if found
    >>> cval.found == True
    >>> cval.val # the value for 'CVAL from the header
    >>> # if not found
    >>> cval.found == False
    >>> cval.val == no_val
    
    """
    # this class records the header information if it's found and provides a simple way to check 
    def __init__ (self,header, keyword, noval = 0, verbose=False):
        self.keyword = keyword
        if keyword in header.keys():
            self.val = header[keyword]
            self.found = True
        else:
            verbose = bool(verbose)
            self.val = noval
            if verbose: print "keyword:",keyword,"not found"
            self.found = False
    
    def __repr__ (self):
        return self.val

pass
# ############################################################################# #
# classes to store the coefficients and wavelength solution as well as to solve
class WavelengthSolutionFunctions:
    """
    A class which holds the wavelength solutions
    
    Parameters
    ----------
    None
        
    Returns
    -------
    None    
    
    Attributes
    ----------
    None
    
    
    Notes
    -----
    __1)__ Note about some of the inputs below
    
    
    
    Examples
    --------
    >>>
    >>>
    >>>
    
    """

    def __init__ (self):
        ft = {'no solution'    : ["wl = np.arange(start_pt,start_pt+len(pts))",self.wl_soln_no_solution_prog],
              'none'           : ["same as 'no solution'",self.wl_soln_no_solution_prog],
              None             : ["same as 'no solution'",self.wl_soln_no_solution_prog],
              'pts'            : ["wl = pts",self.wl_soln_pts],
              'ordinary poly'  : ["wl = fxn(pts)  where fxn is an ordinary 7th degree polynomial",self.wl_soln_ordinary_poly],
              'poly'           : ["same as 'ordinary poly'",self.wl_soln_ordinary_poly],
              'chebyshev poly' : ["wl = fxn(pts) where fxn is a Chebyshev polynomial",self.wl_soln_chebyshev_poly],
              'spline3'        : ["wl = fxn(pts) where fxn is a cubic spline function",self.wl_soln_cubic_spline],
              'legrandre poly' : ["wl = fxn(pts) where fxn is a Legrandre polynomial",self.wl_soln_legrandre_poly],
              'linear'         : ["wl = fxn(pts) where fxn is a polynomial of order 1, uses coeff[0],coeff[1], defauts to coeff[1] = 1 (i.e. no dispersion)",self.wl_soln_linear],
              'log linear'     : ["wl = 10**(fxn(pts)) where fxn is the same polynomial described in 'linear'",self.wl_soln_log_linear_2_linear]}
        self._function_types = ft
    
    def __call__ (self, pts, coeff, function_type='no solution', default_function=True):
        """
        convert points to wavelength in Angstroms via the wavelength solution
        
        INPUT:
        pts : array, contains the points used to convert element-by-element to wavelength
        coeff : array len(coeff) > 2, contains the dispersion coefficients to use in the polynomial to the wavelength solution
        function_type :  can specify a specific solution to be applied
                    'no solution'    : wl = np.arange(disp[0],disp[0]+len(pts)), can use disp[0] to advance where the starting point is
                    'none' or None   : same as 'no solution'
                    'pts'            : wl = pts
                    'ordinary poly'  : wl = fxn(pts)  where fxn is an ordinary 7th degree polynomial
                    'poly'           : same as 'ordinary poly'
                    'chebyshev poly' : wl = fxn(pts) where fxn is a Chebyshev polynomial
                    'spline3'        : wl = fxn(pts) where fxn is a cubic spline function
                    'legrandre poly' : wl = fxn(pts) where fxn is a Legrandre polynomial
                    'linear'         : wl = fxn(pts) where fxn is a polynomial of order 1, uses coeff[0],coeff[1], defauts to coeff[1] = 1 (i.e. no dispersion)
                    'log linear'     : wl = 10**(fxn(pts)) where fxn is the same polynomial described in 'linear'
        
        default_function : (bool) If True it will use a default function if not of the above are found
        
        """
        ft = function_type
        if ft == 'no solution': coeff = [0,1]
                
        pts,coeff,no_solution = self._check_pts_coeff(pts,coeff)        
        if no_solution: ft = 'no solution'
        
        if ft not in self._function_types:
            if default_function: ft = 'no solution'
            else:raise ValueError("Unknown function type:"+str(function_type))
        
        if ft is 'pts': return self.wl_soln_pts(pts)
        
        func = self._function_types[ft][1]
        return func(pts,coeff)

    def get_function_types (self):
        return (self._function_types.keys())
    
    def get_func_name (self,name):
        if name not in self._function_types: raise ValueError("Unknown function type:"+name)
        else: return name
    
    def _check_pts_coeff (self,pts,coeff):
        no_solution = False
        try: pts = np.array(pts)
        except: no_solution = True
        
        try: coeff = np.array(coeff)
        except: no_solution = True
        
        if not no_solution:
            # check length of dispersion equation
            if len(coeff) < 2:
                print "WARNING: ARRAY coeff MUST BE len(coeff) >= 2, RUNNING coeff = array([0.,1.])"
                coeff = np.array([0.,1.])
                
            if len(coeff) < 8:
                coeff = np.concatenate((coeff,np.zeros(8-len(coeff))))
                
            # check the first order coefficient, if zero there is no solution
                if coeff[1] == 0: no_solution=True
        
        return pts,coeff,no_solution
    
    def wl_soln_no_solution_prog (self,pts,coeff):
        start_pt = 1

        try: coeff = np.array(coeff,dtype=int)
        except: coeff = None

        if str(type(coeff[0])).find('int') != -1:
            start_pt = int(coeff[0])

        return np.arange(start_pt,start_pt+len(pts))

    def wl_soln_no_solution (self,pts):
        return self.wl_soln_no_solution_prog(pts,None)

    def wl_soln_pts (self,pts):
        return pts

    def wl_soln_log_linear_2_linear (self,pts,coeff):
        return 10**(coeff[0] + coeff[1]*pts)
    
    def wl_soln_linear (self,pts,coeff):
        return coeff[0] + coeff[1]*pts
    
    def wl_soln_legrandre_poly (self,pts,coeff):
        xpt = (2.*pts - (len(pts)+1))/(len(pts)-1)
        wl = coeff[0] + coeff[1]*xpt
        return wl

    def wl_soln_cubic_spline (self,pts,coeff):
        print "!! DOUBLE CHECK SPLINE SOLUTION"
        # This matches what spectre gives but it seems like it give redundant solutions, i.e. all wl are the same
        s= (pts - 1)/(len(pts)-1) * coeff[7]
        J = np.array(s,dtype=int)
        a = (J+1) - s
        b = s - J
        z0 = a**3
        z1 = 1. + 3.*a*(1.+a*b)
        z2 = 1. + 3.*b*(1.+a*b)
        z3 = b**3
        c  = [coeff[x] for x in J]
        c1 = [coeff[x+1] for x in J]
        c2 = [coeff[x+2] for x in J]
        c3 = [coeff[x+3] for x in J]
        wl = c*z0 + c1*z1 + c2*z2 + c3*z3
        return wl 
    
    def wl_soln_chebyshev_poly (self,pts,coeff):
        if len(coeff) < 4: 
            raise ValueError("Number of coefficients insufficent for Chebyshev")
        #c20    p = (point - c(6))/c(7)
        #c      xpt = (2.*p-(c(9)+c(8)))/(c(9)-c(8))
        # !! is this right?
        xpt = (2.*pts - (len(pts)+1.))/(len(pts)-1.) 
        
        # xpt = (2.*point-real(npt+1))/real(npt-1)

        wl =  coeff[0] + xpt*coeff[1] + coeff[2]*(2.0*xpt**2.0-1.0) + coeff[3]*xpt*(4.0*xpt**2.0-3.0)+coeff[4]*(8.0*xpt**4.0-8.0*xpt**2.0+1.0)
        return wl
    
    def wl_soln_ordinary_poly (self,pts,coeff):
        # maximum degree polynomial will be determined
        # as degree 7
        degree = 7
        coeff = coeff[:degree]
        coeff = list(coeff)
        # must reverse my coeff
        coeff.reverse()
        return scipy.polyval(coeff,pts)
    
wavelength_solution_functions = WavelengthSolutionFunctions()

class WavelengthSolutionCoefficients:
    """
    This class holds the coefficients as extracted from the fits header
    """
    
    def __init__ (self, equ_type='none' , extra='none'):
        """
        equ_type correpsonds to the WavelengthSolutionFunctions
        coeffs correponds to the function but for a polynomial
             coeffs = c
             y = c[0]+c[1]*x+c[2]*x**2+c[3]*c**3+....
        extra     extra info from the extraction process
        """
        self.equ_type = self.set_equation_type(equ_type)
        self.coeffs = []
        self.extra = str(extra)
        self._rv = 0.0
    
    def __len__ (self):
        return len(self.coeffs)
    
    def add_coeffs (self,coeff,index=None):
        if index is None: 
            self.coeffs.append(coeff)
        else: 
            self.coeff[index] = coeff
        
    def get_coeffs (self,index=None):
        if index not in xrange(len(self.coeffs)): 
            index = len(self.coeffs)-1
        return self.coeffs[index]
    
    def get_equation_type (self):
        return deepcopy(self.equ_type)
    
    def set_equation_type (self,equ_type):
        self.equ_type = wavelength_solution_functions.get_func_name(equ_type)
    
    def get_extra_info (self):
        return deepcopy(self.extra)

    def set_rv (self,rv):
        self._rv = float(rv)
        
    def get_rv (self):
        return self._rv

pass
# ############################################################################# #
# misc functions which could be used
def pts_2_phys_pixels (pts,bzero=1,bscale=1):
    """
    convert points to wavelength in Angstroms via the wavelength solution
    INPUT:
    pts : array, contains the points used to convert element-by-element to wavelength
    
    bzero : float, from the header file which gives the starting point for the physical pixels
    bscale : float, from the header file which gives the scaling for the physical pixels
            pts = bzero + pts*bscale
    
    
    bzero = query_fits_header(header,'BZERO',noval=1) # for scaled integer data, here is the zero point
    bscale = query_fits_header(header,'BSCALE',noval=0) # for scaled integer data, here is the multiplier
    
    I'm pretty sure fits uses these when it reads in the data so I don't need to
   
    """
    if bzero != 0 or bscale !=1:
        print "Whoops, I don't know exactly what to do with bzero!=1 or bscale!=0 :<",bzero,"><",bscale,">"
        #pts +=1
    
    pts = bzero + pts*bscale

    # should return [1,2,3,......,#pts]
    return pts

def check_for_txt_format (filename,**np_kwargs):
    try: txt_data = np.loadtxt(filename,unpack=True,dtype=float,**np_kwargs)
    except: return False, None
    return True, txt_data

pass
# ############################################################################# #
# functions to extract wavelength solution coefficients and equation type from
# a fits header

class ExtractWavelengthCoefficients:
 
    def __init__ (self,fits_header):
        # TODO: type check fits_header
        self.header = fits_header
        
        # This is the order to resolve the coefficients in
        self.resolve_coeffs = OrderedDict()
        self.resolve_coeffs['spectre'] = self.from_SPECTRE
        self.resolve_coeffs['wv_0'] = self.from_makee_wv
        # self.resolve_coeffs['co_0'] = self.from_makee_co
        self.resolve_coeffs['w0'] = self.from_w0
        self.resolve_coeffs['wcs'] = self.from_wcs
        self.resolve_coeffs['crvl'] = self.from_crvl
        self.resolve_coeffs['ctype1'] = self.from_ctype1
        self.resolve_coeffs['linear'] = self.from_pts 
        self.resolve_coeffs['no solution'] = self.from_none           
     
    def __repr__ (self):
        output = "ExtractingWavelengthCoefficients\n"
        output += repr(self.header)
        return output

    def __iter__ (self):
        return iter(self.resolve_coeffs)
      
    def __getitem__ (self,coeff_type):
        if coeff_type not in self.resolve_coeffs:
            raise KeyError("Unknown coeff type")
        return self.resolve_coeffs[coeff_type]
      
    def get_coeff (self,preferred=None,all_=False):
        if preferred is not None:
            try: 
                return self[preferred]
            except KeyError:
                print "Unknown preferred type"

        cc = []
        for coeff_type in self:
            wlcoeff = self[coeff_type]
            if len(wlcoeff) == 0: 
                continue
            
            if not all_:
                return wlcoeff
        
            cc[coeff_type] = wlcoeff
        return cc 
    
    @property
    def from_ctype1 (self):
        """
        Finds keywords CTYPE1, CRVAL1, CRPIX1, and CDELT1 and extracts the
        coefficients for the linear wavelength solution
        """
        wlcoeff = WavelengthSolutionCoefficients()
        #==========================================================================#
    
        ctype1 = query_fits_header(self.header,'CTYPE1',noval='')
        crval1 = query_fits_header(self.header,'CRVAL1',noval=1) # for linear dispersions, the starting wavelength
        crpix1 = query_fits_header(self.header,'CRPIX1',noval=1) # for linear dispersions, the pixle to which CRVAL1 refers
        cdelt1 = query_fits_header(self.header,'CDELT1',noval=1) # for linear dispersion, here is the dispersion

        start_pix = crpix1.val+1 # this is because I start the pixel counting at 1 later
        if (ctype1.found and ctype1.val == 'LINEAR'):
            if crval1.found and crpix1.found and cdelt1.found:
                coeff = np.array([crval1.val-start_pix*cdelt1.val, cdelt1.val])
                wlcoeff.extra = 'used header to get crval1 and cdelt1, to apply wl = crval1 + cdelt1*pts'
                wlcoeff.set_equation_type('linear')            
                wlcoeff.add_coeffs(coeff)
    
        if (ctype1.found and ctype1.val == 'LOG-LINEAR'):
            if crval1.found and crpix1.found and cdelt1.found:
                coeff = [crval1.val-start_pix*cdelt1.val,cdelt1.val]
                wlcoeff.extra = 'LOG-LINEAR, used header to get crval1 and cdelt1, to apply wl = 10**(crval1 + cdelt1*pts)'
                wlcoeff.set_equation_type('log linear')            
                wlcoeff.add_coeffs(coeff)
                  
        return wlcoeff

    @property
    def from_crvl (self):
        """
        Looks at all the CRVL1_?? and CDLT1_?? keywords where ?? is all the possible orders
        (e.g. order 1 ==> CRVL1_01) 
        
        This looks at the LININTRP keyword but only 'linear' is accepted.
        
        """
        
        wlcoeff = WavelengthSolutionCoefficients()
        
        def get_for_order (ordi):
            ordi = format(ordi,'02')
            #==========================================================================#
            linintrp = query_fits_header(self.header,'LININTRP') # string with infor about the type of linear interpretation
            crvl1_ = query_fits_header(self.header,'CRVL1_'+ordi,noval=1) # for linear dispersions, the starting wavelength
            cdlt1_ = query_fits_header(self.header,'CDLT1_'+ordi,noval=0) # for linear dispersions, the pixle to which 
            
            if crvl1_.found and cdlt1_.found:
                if linintrp.found and linintrp.val.find('linear') == -1: 
                    print "WARNING: KEYWORD LININTRP HAS NO REFERENCE TO 'linear' BUT PERFORMING A LINEAR DISPERSION"
                coeff = [crvl1_.val, cdlt1_.val]
                wlcoeff.extra = 'used header to get crvl1_ and cdlt1_ depending on order, to apply wl = crvl1_? + cdlt1_?*pts'
                wlcoeff.set_equation_type('linear')
                wlcoeff.add_coeffs(coeff)
                return False
            else: 
                if ordi == 0: 
                    return False
                return True
                
        for i in xrange(100):
            if get_for_order(i): 
                break
          
        if len(wlcoeff) > 0: 
            print "!! FIRST TIME WITH CRVL AND CRDEL, CHECK THE OUTPUT OF WAVELENGTH VS FLUX"
            
        return wlcoeff

    @property
    def from_wcs (self): 
        """
        This extracts the wavelength solution from IRAF's World Coordinate System
        It is limited in coordinates
        
        """ 
        wlcoeff = WavelengthSolutionCoefficients()
    
        #==========================================================================#
        wat0_001 = query_fits_header(self.header,'WAT0_001')
        wat1_001 = query_fits_header(self.header,'WAT1_001') 
        # wat2_001 = query_fits_header(self.header,'WAT2_001') 
        wat3_001 = query_fits_header(self.header,'WAT3_001') 
    
        if not wat0_001.found: return wlcoeff
        # found the wat0_001 keyword
        
        wat0 = wat0_001.val.split("=")
        if wat0[0].lower() != 'system': 
            raise ValueError("IRAF WCS, Unknown keyword in WAT0_001 = "+wat0[0])
    
        wcs_system = wat0[1]
        if wcs_system not in ('physical','multispec'): 
            # print "IRAF WCS: Unknown system given in WAT0_001 = "+wcs_system
            return wlcoeff
        
        # can't be 'world' or 'equispec' because those are for 2D data not 1D
        if wat3_001.found: 
            print "HeadsUp: CHECK OUT WAVELENGTH DISPERSION, FOUND KEYWORD 'WAT3_001' WHICH I DON'T KNOW HOW TO DEAL WITH" 
        # similarly, wat3 corresponds to higher dimensions than expected
        
        if not wat1_001.found: 
            return wlcoeff
            
        #========== now it is on system mutlispec
        wat1_001_dict = {}
        for val in wat1_001.val.split():
            val = val.split("=")
            wat1_001_dict[val[0]] = val[1]
    
        # WAT1_001= 'wtype=linear label=Wavelength units=Angstroms' 
        # WAT1_001= 'wtype=multispec label=Wavelength units=angstroms'
        
        # check wtype:
        if wat1_001_dict['wtype'] == 'linear': 
            raise ValueError("IRAF WCS keyword WAT1_001 has wtype = linear, I don't know how to deal with this")
        elif wat1_001_dict['wtype'] != 'multispec': 
            raise ValueError("Unknown value for WAT1_001 wtype:"+wat1_001_dict['wtype'])
       
        # check wavelength
        if wat1_001_dict['label'].lower() != 'wavelength': 
            raise ValueError("IRAF WCS for WAT1_001 keyword I expected label=wavelength but got label="+str(wat1_001_dict['label']))
        
        # check units
        if wat1_001_dict['units'].lower() != 'angstroms': 
            raise ValueError("ERROR: DON'T KNOW HOW TO HANDLE IRAF WAT1_001 FOR UNITS GIVEN"+wat1_001_dict['units'])
     
     
        #======== now has 'wtype=multispec label=Wavelength units=angstroms'        
        
        #        ordi = format(ordi,'02')
        #        wat2_0 = query_fits_header(self.header,'WAT2_0'+ordi,'02'))
        #        if not wat2_0.found: return True
        #    
    
        def unpack_WCS_multispec (root_keyword):     
            # creates order_disp which is a dictionary that has the dispersion for a given order, this would be inefficient to make the whole thing every time when you only need one value every time it's run. But with the way IRAF layed out these WAT2 stuff and how my code is written it's sort of necessary
            wat_str = ''
            for key in self.header.keys():
                if key.find(root_keyword) == 0: wat_str += format(self.header[key],'68')
            
            cut_str =  wat_str[:30].split()
            if cut_str[0].lower() != 'wtype=multispec':
                raise ValueError("IRAF WCS: For root keyword "+root_keyword+" didn't get wtype=multispec :"+wat_str[:30])
                
    
            # separate lines by spec 
            sep_spec = wat_str[15:].split("spec")
            order_coeff = {}
            def _check_string_for_splits (dat,k,ret = 'prefix'):
                """
                This is used to break up values like:
                4.21234D+50.234    =>  4.21234E+5  and  0.234
                -2.12345E-40.234  => -2.12345E-4  and  0.234
                2.3451650.456     => -2.345165    and  0.456
                2.123-32.123      =>  2.123       and  -32.123
                2.124E-4-32.123   =>  2.123E-4    and  -32.123
    
                can't deal with:
                +2.345   and   +23.4      => +2.345+23.4
                2.123    and   +32.12     =>  32.123+32.123
                
                sdat is the line split by '.'
    
                ret = 'prefix' or 'suffix'
                """
                sdat = dat.split('.')
                #>> deal with the first case, assume sdat[0] is fine
                # check for scientific notation
                sdat[k].upper()
                if sdat[k].find('D') != -1: sdat[k] = sdat[k].replace('D','E') # change D to E
                if sdat[k].find('+') != -1 and sdat[k].find('E+') == -1: print "I didn't code anything for '+', check the output. ORDER:"+cur_order+" problem line:"+dat
    
                checkfor = ['E-','E','E+']
                found_checkfor = False
                for cval in checkfor:
                    if sdat[k].upper().find(cval) != -1: 
                        found_checkfor = True
                        ssdat = sdat[k].split(cval) # assume ssdat[0] if fine (e.g. 2345E-40 => ['2345','40])
                        if cval == 'E': cval = 'E+'
                        ss_dat = ssdat[1].split('-')
                        if len(ss_dat) == 1: # (e.g. '-23.2345E-40.23' =1> '2345E-40' => ['2345','40] => [40])
                            if ss_dat[0][-1] != '0' and ret == 'prefix' and k!=0: raise ValueError("ORDER:"+cur_order+" I expected a zero, bad input:"+dat)
                            else:
                                suffix = ssdat[0]+cval+ss_dat[0][:-1] # (e.g. 2345E-40 => 2345E-4)  !! what if I wanted 2345E-10
                                prefix = '0'
                        elif len(ss_dat) == 2: # (e.g. 2345E-4-34 => ['2345','4-34] => ['4','34'])
                            suffix = ssdat[0]+cval+ss_dat[0]
                            prefix = "-"+ss_dat[1]
                        else: # (e.g. 2345E-4-3-24 => ['2345','4-3-24] => ['4','3','24'])
                            raise ValueError("ORDER:"+cur_order+" unexpected value has multiple dashes/negative signs: "+dat)
                        break
                # if it's not scientific notation
                if not found_checkfor:
                    ss_dat = sdat[k].split('-')
                    if len(ss_dat) == 1: # (e.g. '234540' => ['234540'])
                        if ss_dat[0][-1] != '0' and ret == 'prefix' and k!=0: raise ValueError("ORDER:"+cur_order+" I expected a zero, bad input:"+ss_dat[0])
                        else:
                            suffix = sdat[k] # (e.g. 234540 => 234540) 
                            prefix = '0'
                    elif len(ss_dat) == 2: # (e.g. '2345-34' => ['2345','34])
                        suffix = ss_dat[0]
                        prefix = "-"+ss_dat[1]
                    else: # (e.g. 2345-3-24 => ['2345','3','24'])
                        raise ValueError("ORDER:"+cur_order+" unexpected value has multiple dashes/negative signs: "+dat)
                if ret == 'prefix': return prefix
                elif ret == 'suffix': return suffix
                else: raise TypeError("Whoops, I expected either 'prefix' or 'suffix'")
    
            for val in sep_spec:
                # split by the equals order_num = stuff
                sep_spec2 = val.split("=")
                
                is_problem_child = False
                debug_problem_children = False
                if len(sep_spec2)>1:
                    cur_order = str(sep_spec2[0])
                    sep_spec_val = sep_spec2[1].replace('"','').split()
                    # go through and make sure no values have extra decimal (e.g. -0.0040.234) which happened for one fits file when the wavelength solution was given twice
                    for i in range(len(sep_spec_val)):
                        dat = deepcopy(sep_spec_val[i])
    
                        sdat = dat.split(".")
                        new_entries = []
                        if len(sdat) > 2:
                            if debug_problem_children: print "=== "+cur_order+" ===== problem child ===>",dat
                            is_problem_child = True
                            #prefix,suffix = _check_string_for_splits (dat,1) 
                            #new_entries.append(sdat[0]+"."+suffix)
    
                            for j in range(1,len(sdat)): # len>2
                                prefix = _check_string_for_splits (dat,j-1,ret='prefix')
                                suffix = _check_string_for_splits (dat,j,ret='suffix')
                                new_entries.append(prefix+"."+suffix)
    
                        # now stick in new_entries
                        if len(new_entries) != 0:
                            new_entries.reverse()
                            del sep_spec_val[i]
                            for new_entry in new_entries:
                                sep_spec_val.insert(i,new_entry) 
    
                    if is_problem_child and debug_problem_children:
                        for i in range(len(sep_spec_val)):
                            print i,">>",sep_spec_val[i]
                        skip = raw_input("")
                        if skip == 'a': import pdb; pdb.set_trace()
                        
                    order_coeff[int(sep_spec2[0])] = np.array(sep_spec_val,dtype=float)
            return order_coeff, wat_str
        
        order_coeff, wat2_str = unpack_WCS_multispec('WAT2_')
        # REFERNCE: http://iraf.net/irafdocs/specwcs.php
        # The dispersion functions are specified by attribute strings with the identifier specN where N is the physical image line. The attribute strings contain a series of numeric fields. The fields are indicated symbolically as follows. 
        #     specN = ap beam dtype w1 dw nw z aplow aphigh [functions_i]
        # 
        # order_coeff[?][0] = aperture number
        # order_coeff[?][1] = beam number
        # order_coeff[?][2] = dispersion type, dcflag = -1 no disp, 0 linear, 1 log-linear, 2 nonlinear
        # order_coeff[?][3] = c0, first physical pixel
        # order_coeff[?][4] = c1, average disperasion interval
        # order_coeff[?][5] = npts, number valid pixels !! could have problem if different orders have different lengths
        # order_coeff[?][6] = rv,z, applies to all dispersions coordinates by multiplying 1/(1+z)
        # order_coeff[?][7] = aplow, lower limit of aperture
        # order_coeff[?][8] = aphigh, upper limit of aperture
        # order_coeff[?][9] = N/A for linear or log-linear
        # ----------------
        #            function_i =  wt_i w0_i ftype_i [parameters] [coefficients]
        # order_coeff[?][9]  = wieght wt_i 
        # order_coeff[?][10] = zeropoint offset w0_i
        # order_coeff[?][11] = type dispersion fxn = 1 cheby, 2 legrandre, 3 cubic spline3, 
        #                        4 linear spline, 5 pixel coordinate array, 6 sampled coordinate array
        # order_coeff[?][12+] = [parameters...]
        # order_coeff[?][12++] = [coefficients...]
    
        if 1 not in order_coeff: 
            raise ValueError("IRAF WCS: I didn't find a spec1:"+wat2_str[:40])
        dcflag = order_coeff[1][2]
        equ_type = 'none'
        
        LTV_dimn = [1,1]
      
        def get_order_wlsoln (ordi):
            if ordi not in order_coeff: 
                if i == 0: return False
                return True
            # w = sum from i=1 to nfunc {wt_i * (w0_i + W_i(p)) / (1 + z)}
            if order_coeff[ordi][2] != dcflag: 
                raise ValueError("Encountered orders with different functions dcflag1="+str(dcflag)+"  dcflag"+str(ordi)+"="+str(order_coeff[ordi][2]))
            z = order_coeff[ordi][6]
            if z != 0: 
                rv = 1./(1.+z)
                if wlcoeff.get_rv() != 0.0 and rv != wlcoeff.set_rv:
                    # TODO : allow two orders to have different rv values
                    raise StandardError("Whoops, this means that two orders have different rv values")
                wlcoeff.add_rv(rv)
        
            if dcflag < 0:
                equ_type='none'
                coeff = [0,1]
            
            elif dcflag == 0:
                equ_type = 'linear'
                coeff = [order_coeff[ordi][3],
                         order_coeff[ordi][4]]
                
            elif dcflag == 1 or dcflag == 2:
                polytype = order_coeff[ordi][11]
                ltv = query_fits_header(self.header,'LTV'+str(LTV_dimn[0]),noval=0) # IRAF auxiliary wavelenth solution parameters 
                ltm = query_fits_header(self.header,'LTM'+str(LTV_dimn[0])+'_'+str(LTV_dimn[0]),noval=0) # IRAF auxiliary wavelenth solution parameters 
            
                if (ltv.found or ltm.found): 
                    print ("IRAF WCS: found WAT keywords with system=multispec and found dcflag = "+str(dcflag)+" for order "+str(ordi)+" and LTV and LTM keywords which I don't know what to do with")
            
                if polytype == 1:
                    # order_coeff[?][11] = type dispersion fxn = 1 cheby
                    # order_coeff[?][12] = order
                    # order_coeff[?][13] = xxmin
                    # order_coeff[?][14] = xxmax
                    # order_coeff[?][15:15+order] = [coefficients...]
            
                    # !! the points used to calculate the wavelength solution: np.arange(xxmin,xxmax)
                    xxmin = order_coeff[ordi][13]
                    if xxmin != 1: 
                        print "WARNING: From WCS I got a xxmin of",xxmin,"but hardwired in is a value of 1"
            
                    equ_type = 'chebyshev poly'
                    poly_order = order_coeff[ordi][12]
                    xxmin, _ = order_coeff[ordi][13:15]
                    coeff = order_coeff[ordi][15:15+poly_order] # I think
            
                    #coeff[5] = ltv.val # from SPECTRE == c(6)
                    #coeff[6] = ltm.val # ==c(7)          
                    #c*****a Chebyshev polynomial solution
                    #c20    p = (point - c(6))/c(7)
                    #c      xpt = (2.*p-(c(9)+c(8)))/(c(9)-c(8))
    
    
            elif polytype == 2:
                equ_type = 'legrandre poly'
                print "WARNING: NOT SURE IF I'M GETTING THE LEGRANDRE COEFFICIENTS FROM DATA CORRECTLY"
                coeff = order_coeff[ordi][14:] # I think
    
                #coeff[5] = ltv.val  # from SPECTRE
                #coeff[6] = ltm.val                                
    
            elif polytype == 3: 
                equ_type = 'spline3'
                print "WARNING: NOT SURE IF I'M GETTING THE CUBIC SPLINE COEFFICIENTS FROM DATA CORRECTLY"
                coeff = order_coeff[15:]
                
            elif polytype == 4:
                print "WARNING: I CAN'T CURRENTLY COMPUTE A LINEAR SPLINE, ASSUMING LINEAR DISPERSION"
                equ_type = 'linear'
                coeff = [0,1]
                coeff = [order_coeff[ordi][3],
                         order_coeff[ordi][4]]
            
            elif polytype > 5:
                raise IOError("ERROR: NOT SET UP TO HANDLE THIS TYPE OF DISPERSION"+str(polytype))
    
            # apply the radial velocity shift given in the WCS
            coeff = np.array(coeff)
            
            wlcoeff.set_equation_type(equ_type)
            wlcoeff.add_coeffs(coeff)
            return False
    
        for i in xrange(100):
            if get_order_wlsoln(i): break
            
        wlcoeff.extra = ['used header to get parameters and coefficients, function: '+equ_type+', to apply wl = function(pts)',order_coeff]
        return wlcoeff

    @property
    def from_w0 (self):
        """
        Extracts coefficients from the W0 and WPC keywords
        
        """
        
        wlcoeff = WavelengthSolutionCoefficients()
    
        w0 = query_fits_header(self.header,'W0',noval=0) # for linear dispersions, starting wavelength
        wpc = query_fits_header(self.header,'WPC',noval=0) # for linear dispersion, here is th dispersion
        if w0.found and wpc.found:
            coeff = [w0,wpc]
            wlcoeff.extra = 'used header to get W0 and WPC, to apply wl = W0 + WPC*pts'
            wlcoeff.set_equation_type('linear')
            wlcoeff.add_coeffs(coeff)
            print "!! FIRST TIME WITH W0 AND WPC, CHECK THE OUTPUT OF WAVELENGTH VS FLUX"
        return wlcoeff

    @property
    def from_pts (self):
        wlcoeff = WavelengthSolutionCoefficients()
        wlcoeff.extra = 'no header info used, just a wl = pts'
        wlcoeff.set_equation_type('linear')
        wlcoeff.add_coeffs([0,1])        
        return wlcoeff

    @property
    def from_none (self):
        wlcoeff = WavelengthSolutionCoefficients()
        wlcoeff.extra = 'following checks found that the first order coefficient is zero, setting basic linear wl = pts'
        wlcoeff.set_equation_type('linear')
        wlcoeff.add_coeffs([0,1])        
        return wlcoeff        

    @property
    def from_makee_wv (self):
        """
        Extracts cooefficients from the WV_0_?? and WV_4_?? keywords where ??
        is based on the order (e.g. order 1 = WV_0_01) all orders are extracted
        """
    
        #==========================================================================#
        wlcoeff = WavelengthSolutionCoefficients()
    
        def get_for_order (order_id):
            WV_0_ = query_fits_header(self.header,'WV_0_'+format(int(order_id),'02')) # first 4 coefficients
            WV_4_ = query_fits_header(self.header,'WV_4_'+format(int(order_id),'02')) # second 4 coefficients
            
            if not WV_0_.found: return True
            
            coeff = np.zeros(8)
            coeff[:4] = np.array(WV_0_.val.split(),dtype=float)
            if WV_4_.found: coeff[4:8] = np.array(WV_4_.val.split(),dtype=float)
            else: raise IOError("Expected to get "+'WV_4_'+format(int(order_id),'02')+" along with keyword "+'WV_0_'+format(int(order_id),'02'))
    
            # !! could put a check to make sure WV_4_ follows WV_0_ in the header
            wlcoeff.extra = 'used header to get MAKEE keywords WV_0_? and WV_4_? depending on order, to apply polynomial coefficients given by WV_0_? and WV_4_?'
            wlcoeff.set_equation_type('poly')
            wlcoeff.add_coeffs(coeff)
            return False
                
    
        for i in xrange(1,100):
            if get_for_order(i): break
        
        #==========================================================================#   
        return wlcoeff

    @property
    def from_makee_c0 (self):
        """
        Extracts cooefficients from the CO_0_?? and CO_4_?? keywords where ??
        is based on the order (e.g. order 1 = CO_0_01) all orders are extracted
        
        HOWEVER:
        currently I don't know what to do with these values
        
        """     
        wlcoeff = WavelengthSolutionCoefficients()
        #==========================================================================#
        def get_from_order (order_id):
            CO_0_ = query_fits_header(self.header,'CO_0_'+format(int(order_id),'02')) # first 4 coefficients
            CO_4_ = query_fits_header(self.header,'CO_4_'+format(int(order_id),'02')) # second 4 coefficients
                
            if CO_0_.found or CO_4_.found: 
                print "WARNING: KEYWORDS",'CO_0_'+format(int(order_id),'02'),"AND",'CO_4_'+format(int(order_id),'02'),"FOUND BUT I DON'T KNOW WHAT TO DO WITH THEM"
                
            if CO_0_.found and CO_4_.found:
                coeff = np.zeros(8)
                coeff[:4]  = np.array(CO_0_.val.split(),dtype=float)
                coeff[4:8] = np.array(CO_4_.val.split(),dtype=float)
                wlcoeff.add_coeffs(coeff)
                return False    
            return True
        
        for i in xrange(1,100):
            if get_from_order(i): break
        
        if len(wlcoeff) > 0: 
            wlcoeff.set_equation_type('none')
            wlcoeff.extra = 'coefficients from C0_0_? and C0_4_? makee pipeline keywords'  
             
        return wlcoeff

    @property
    def from_SPECTRE (self):
        """
        Extracts the wavelength solution from the SPECTRE HISTORY tags
        """
        wlcoeff = WavelengthSolutionCoefficients()
        if not query_fits_header(self.header,'HISTORY',noval=0).found: 
            return wlcoeff # old SPECTRE-stype dispersion information

        #==========================================================================#
        spectre_history, _history_lines = self.get_SPECTRE_history()            
        # if you found history lines use one with the most recent data                
        if len(spectre_history) > 0:
            most_recent = sorted(spectre_history.keys())[-1]
                
            extra_data, disp_type, coeff = spectre_history[most_recent][1:]
            
            wlcoeff.extra = extra_data
            wlcoeff.set_equation_type(disp_type)
            wlcoeff.add_coeffs(coeff)
            
        # NOTE: spectre_history has all the history tags for spectre keyed by time stamp                          
        return wlcoeff
        
    def get_SPECTRE_history (self):
        if True:
            return {},[]
        spectre_history = {}
        kount = 0


        def parse_spectre_history (hcard):
            
            # cut up the line
            date_str, dstr = hcard[:10], hcard[11:17], 
            coefficients = hcard[18:37], hcard[37:54], hcard[54:] 

            # get the coefficients 

            coefficients = [float(s.replace('D','e')) for s in coefficients]
            day,month,year = [int(s) for s in date_str.split(':')]
            timetag = time.mktime((year,month,day,0,0,0,0,0,0))

            return timetag, dstr, coefficients


        histories = self.header['HISTORY']
        for hcard in histories:

            timetag, dstr, coefficients = parse_spectre_history(hcard)
            co_tags = hcard[11:17]
                        
                

            
        
        





        for i in xrange(len(histories)):
            line=str(histories[i]).rstrip()
            history_lines.append(line)
            line = "HISTORY "+line
            if line[23:28] == 'DISP=':
                raise StandardError("!! NEED TO RESOLVE HOW TO READ THIS TYPE OF DISPERSION FROM SPECTRE")                    
            #                   read (head(k+1:k+80),1022) (disp(i),i=1,4)
            # 1022              format(28x,1p4e13.5)
            
            if line[19:26] == 'D1,2,3:':
                sline = line.split(":")
                day = int(sline[0].split()[1])
                month = int(sline[1])
                year = int(sline[2].split()[0])
                timetag = time.mktime((year,month,day,0,0,kount,0,0,0))
                kount += 1
            
                coeff = np.zeros(9)
                line = line.replace('D','e')
                coeff[:3] = np.array([line[26:44],line[44:62],line[62:80]],dtype=float)
    
                line2= str(histories[i+1]).rstrip().replace('D','e')
                if line2[0:8] != 'HISTORY ': 
                    raise IOError('EXPECTED NEXT LINE TO HAVE TAG HISTORY')
                coeff[3:6] = np.array([line2[26:44],line2[44:62],line2[62:80]],dtype=float)
    
                line3= str(histories[i+2]).rstrip().replace('D','e')
                if line3[0:8] != 'HISTORY ': 
                    raise IOError('EXPECTED NEXT LINE TO HAVE TAG HISTORY')
                disp_info = np.array([line3[26:44],line3[44:62],line3[62:80]],dtype=float)
    
                # cheby poly may need disp_info[0]
                # disp_info[0] == c(7)
                # disp_info[1] == c(8)
                # from SPECTRE: 
                # c20    p = (point - c(6))/c(7)
                # c      xpt = (2.*p-(c(9)+c(8)))/(c(9)-c(8))
                if disp_info[2] == 1: disp_type = 'chebyshev poly'
                elif disp_info[2] == 2: disp_type = 'legrendre poly'
                elif disp_info[2] == 3:
                    print "WARNING: check the output, I think I may need to use disp_info[1] "
                    # from SPECTRE: s = (point-1.)/(real(npt)-1.)*c(8)
                    # c(8) == disp_info[1] ==> true
                    disp_type = 'spline3'
                else: disp_type = 'poly' # but likely the orders > 1 have coefficients zero
                extra_data= ['used header to get SPECRE HISTORY tags, function:'+disp_type+', to apply wl=function(pts)',[line,line2,line3]]
                spectre_history[timetag] = (line,extra_data,disp_type,coeff)
        return spectre_history, history_lines




################################################################
   
pass
# ############################################################################# #
# TODO: The functions below are incomplete until we have the exact 
#     format for the measurementment list and information stuff

def read_txt (filename, get_data=False,**np_kwargs):
    """
    Readin text files with wavelength and data columns (optionally inverse varience)
    
    Parameters
    ----------
    filename : string
        Gives the path to the text file
    get_data : boolean
        If 'True' then it will just return wavelengths, flux, and inv_var
        Otherwise returns a Spectrum object
    np_kwargs : dictionary
        Contains keywords and values to pass to np.loadtxt
        This includes things such as skiprows, usecols, etc
    
    Returns
    -------
    spectrum : Spectrum object 
        If get_data is False returns a Spectrum object
    OR
    data : wavelengths, flux, inv_var
        if get_data is True then returns a tuple of numpy arrays
        
    Raises
    ------
    TypeError : If the value of arg1 isn't 42
    
    
    Notes
    -----
    __1)__ Keywords txt_data, unpack, and dtype are forced for the
        np_kwargs.
    
    
    
    Examples
    --------
    >>>
    >>>
    >>>
    
    """ 
    #### check if file exists   ####### #############
    if not os.path.isfile(filename): 
        raise IOError("File does not exist:'"+filename+"'")

    info = Information()
    info['filename'] = os.path.abspath(filename)

    # Allows for not repeating a loadtxt
    if 'txt_data' in np_kwargs: 
        txt_data = np_kwargs['txt_data']
    else: 
        txt_data = None
    
    if txt_data is None:
        if 'unpack' in np_kwargs: 
            del np_kwargs['unpack']
        if 'dtype' in np_kwargs: 
            del np_kwargs['dtype']
        txt_data = np.loadtxt(filename,unpack=True,dtype=float,**np_kwargs)
    
    # check the input txt_data
    if txt_data.ndim == 1:
        print "HeadsUp: NO WAVELENGTH DATA FOUND, USING FIRST COLUMN AS DATA"
        data = txt_data 
        wls = np.arange(len(data))+1
        var = np.ones(len(data))
    elif txt_data.ndim == 2:
        wls = txt_data[0]
        data = txt_data[1]
        var = np.ones(len(data))
    elif txt_data.ndim == 3: wls,data,var = txt_data
    elif txt_data.shape[0] > 2: 
        print "HeadsUp: Found more than 3 columns in text file '"+filename+"' taking the first three to be wavelength, data, variance respectively"
        wls,data,var = txt_data[:3]
      
    inv_var = var_2_inv_var(var)
    if get_data:
        return (wls,data,inv_var)
    measurement_list = []
    measurement_list.append(Spectrum(wls,data,inv_var))
    
    return measurement_list,info

############################################################################
# readin is the main function for input

def read_fits (filename, hdu=0, band=0, observation_id='', non_std_fits=False):
    """
    Takes a astropy.io.fits hdulist and then for a particular hdu and band
    extracts the wavelength and flux information
    
    This goes through keywords in the header and looks for specific known
    keywords which give coefficients for a wavelenth solution. It then 
    calculates the wavelengths based on that wavelength solution.
    
    
    Parameters
    ----------
    hdulist : astropy.io.fits.HDUList
        A header unit list which contains all the header units
    hdu : integer
        Which hdu from the hdulist to use
    band : integer
        If the hdu has NAXIS3 != 0 then this will select which
        value that dimension should be
    get_data : boolean
        If 'True' then return tuple of (wl,data,inv_var) else return Spectrum
        
    Returns
    -------
    measurement_list or (wl,data,inv_var) : SpectralMeasurementList \
                                            OR tuple of numpy arrays
        Depending on the value of get_data this will return either
        * SpectrumMeasurementList object 
        * a tuple of numpy arrays corresponding to wavelength, flux, inv_var

    Raises
    ------
    IOError : If it encounters unknown KEYWORD options when looking
        for a wavelength solution
    
    Notes
    -----
    none
    
    Examples
    --------
    >>>
    >>>
    >>>
    
    """ 
    if not os.path.isfile(filename): 
        raise IOError("File does not exist:'"+filename+"'")
    #### now check how it behaves as a fits file
    if non_std_fits: 
        hdulist = fits.open(filename)
    else:
        # give standard fits readin a try
        try: hdulist = fits.open(filename)
        except: 
            raise IOError("PYFITS DOES NOT LIKE THE FILE YOU GAVE ('"+filename+"'), TO SEE WHAT ERROR IT GIVES TRY: hdulist = fits.open('"+filename+"')")
    # TODO: check if hdulist is a binary fits table or not and then call correct function
    return _core_read_fits(hdulist,hdu,band,observation_id)
  
def _core_read_binary_fits (hdulist,observation_id=''):
    # TODO: possibly have this also check if the hdulist is for a binary fits table
    # and then check the keywords of the table match 'wl','wavelength','WAVELENGTH'
    # 'FLUX','flux' etc and then read data in that way
    pass   
    
def _core_read_fits (hdulist, hdu=0, band=0, observation_id='', get_data=False):
    #     preferred_wlsoln=None 
    #     # !! should also be able to input wavelength solution?
    #     
    #     if preferred_wlsoln is not None: 
    #         preferred_wlsoln = wavelength_solution_functions.get_func_name(preferred_wlsoln)
    # 
    
    if len(hdulist) > 1: 
        hdu = int(hdu)
        hdu = np.clip(hdu,0,len(hdulist)-1)
    else: 
        hdu = 0

    # specify the current header unit
    header_unit = hdulist[hdu]
    header = header_unit.header

    apformat = query_fits_header(header,'APFORMAT')
    if apformat.found: 
        # TODO: though I think it's just the spec files
        print "WARNING: I'M NOT SURE HOW TO DEAL WITH APFORMAT VALUES" 

    # bzero = query_fits_header(header,"BZERO",noval=0)
    # bscale = query_fits_header(header,"BSCALE",noval=1)

    ###### read in data ##############################################
    data = header_unit.data
    
    # if there's no data return empty
    if data is None:
        wl, data, inv_var = np.zeros(3).reshape((3,1))
        if get_data: 
            return (wl,data,inv_var)
        else: 
            return Spectrum(wl,data,inv_var)
    
    # hdu selection, band select, orders, pixels
    while data.ndim > 3:
        data = data[0]
    if data.ndim == 3:
        # then there are bands
        try: 
            data = data[band]
        except IndexError:
            raise IndexError("Band "+str(band)+" out of range of the data")
    elif data.ndim == 1:
        data = data.reshape((1,-1))
    
    # now the data is ndim==2 with the first dimension as the orders
    # and the second being the data points
    
    ##### Calculate the wavelengths for the data
    # set up wavelength and inverse_variance
    wv = np.ones(data.shape)
    
    # get the wavelength coefficients
    extract_wvcoeffs = ExtractWavelengthCoefficients(header_unit.header)
    # TODO: allow user to edit which wavelength solution to use
    preferred=None
    wvcoeff = extract_wvcoeffs.get_coeff(preferred)

    # TODO: make it possible to select specific order to read in
    orders = xrange(data.shape[0])
    
    # go through all the orders
    do_progress = True
    progressive_pt = 1 # this will advance and be used when there is no wavelength solution
    
    for i in orders:
        # get the coefficients and function type    
        equ_type = wvcoeff.get_equation_type()
        if equ_type in ('none',None,'no solution') and do_progress: 
            coeff = [progressive_pt,1]
            equ_type = 'pts'
        else: 
            coeff = wvcoeff.get_coeffs(i)
            
        # pts[0] = 1 :: this was definitely the right thing to do for SPECTRE's 1-D output but may not be for other equations, may need pts[0]=0,  this may be for bzero,bscale
        pts = np.arange(len(wv[i]))+1
        # apply function
        wv[i] = wavelength_solution_functions(pts, coeff, equ_type)    
        progressive_pt += len(pts)
    
    #=================================================================#
    # return the data .OR. go on and create the spec_obj
    if get_data: 
        return (wl, data, inv_var)
    
    #=================================================================#
    info = Information()
    info['filepath']=os.path.abspath(hdulist.filename())
    info['hdu_used']=hdu
    info['band_used']=band
    info['wavelength_coeffs'] = wvcoeff
    for i,hdu in enumerate(hdulist):
        info['header_'+str(i)]=hdulist[i].header    
    
    rv = wvcoeff.get_rv() 
    measurements = []
    for i in orders:
        wl_soln = WavelengthSolution(wv[i],rv=rv)
        meas = Spectrum(wl_soln,data[i],observation_id=observation_id,order_id=i)
        measurements.append(meas)
    return measurements,info

_core_read_fits.__doc__ = read_fits.__doc__

def read_star (filename):
    """
    Readin a specified Star object save
    """
    # TODO: write this
    pass

pass
# ############################################################################# #
# This is the general read in function for fits and text files

def read (filename,**kwargs):
    """
    TODO: add doc string
    
    **kwargs are passed either to read_txt or read_fits depending
    """
    f = open(filename,'r')
    header_line = f.readline()
    f.close()
    s = _fits_re.search(header_line)
    if s is None:
        return read_txt(filename,**kwargs)
    else:
        # TODO: add more checks of the fits header_line to see
        # if you can parse more about it. e.g. it's an apogee/hst/or/makee file 
        return read_fits(filename,**kwargs)

pass
# ############################################################################# #
# special readin functions

def read_many_fits (filelist,relative_paths=False):
    """
    TODO: fix this doc string
    takes a list of spectre 1D files and creates a single object

    INPUTS:
    =============  =============================================================
    keyword        (type) Description
    =============  =============================================================
    filelist       (string) give the name of a file which contains a list of 
                           spectre files (fits/txt)
                    OR
                   (array) which gives each file name
    relative_paths (bool) if True eyeSpec will look for each file name in the 
                   filelist relative to the current directory. If False it will
                   take the absolute path of the filelist file as the base name
                   to look for each file. Not applicable if filelist is array
    =============  =============================================================
    """
    list_of_files = []
    relative_paths = bool(relative_paths) # !! I think I want to do this every time

    #============================================================#
    # import the files
    #-----------------------------------------------#
    # if given input is a string treat it like it's a file to import
    if isinstance(filelist,(str,np.str_)): 
        # check if file exists
        if not os.path.exists(filelist): 
            raise IOError("Input file not found: '"+filelist+"'")
        
        dirpath = os.path.dirname(os.path.abspath(filelist))+"/"
        f = open(filelist)
        for fname in f:
            fname = fname.rstrip().split()[0]
            if not relative_paths:
                bfname = os.path.basename(fname)
                fname = dirpath+bfname
            if not os.path.exists(fname): 
                raise IOError("File doesn't exist: '"+fname+"'")
            else: 
                list_of_files.append(fname)

        if len(list_of_files) == 0: 
            raise IOError("No valid files found")
        f.close()
    #-----------------------------------------------#
    # if given input is not a string assume it's a list/array
    elif isinstance(filelist,Iterable):
        relative_paths = False
        # see if it's array like and can be used to iterate through
        try: 
            list_of_files = list(np.array(filelist,dtype=str))
        except: 
            raise TypeError("Input must either be a string giving a file which contains lists of files or array like list of file names")

    #============================================================#
    # now with list_of_files import all the objects
    spectral_measurements = []
    all_info = Information()    
    for fname in list_of_files:
        # the code was doing something funny, this is the work around
        if fname[0] == "'" and fname[-1] == "'": 
            fname = fname.replace("'","")
        if not os.path.exists(fname): 
            print("WARNING: not using file '"+fname+"' because it doesn't exist")
            continue
        meas,info = read(fname)        
        spectral_measurements.append(meas)
        all_info[fname] = info
    
    if len(spectral_measurements) == 0: 
        raise IOError("No Data Read")
    return spectral_measurements,all_info

def read_apogee (filename,use_row=1,get_telluric=False,**read_kwargs):
    """ 
    This takes the pipeline reduced fits from the APOGEE. This should contain several header units each with several image extensions.

    Paremeters
    ----------
    filename : string path to fits file
        APOGEE pipeline reduced data with a 0 header unit similar to the below
    use_row : integer 
        APOGEE refers to these as rows, default is row1 ("combined spectrum with individual pixel weighting")
    get_telluric : boolean
        If True then it will also extract the telluric data


    Returns
    -------
    data_meas : SpectralMeasurementList
        A list of all the measurements made
    data_info : Information
        An information dictionary about the data
    tell_meas : SpectralMeasurementList (optional if get_telluric)
        A list of measurements for the Telluric
    tell_info : Information (optional if get_telluric)
        An information dictionary about the telluric
    

    =================================================================
    Example header 0 header unit:
    
    HISTORY APSTAR: The data are in separate extensions:                      
    HISTORY APSTAR:  HDU0 = Header only                                       
    HISTORY APSTAR:  All image extensions have:                               
    HISTORY APSTAR:    row 1: combined spectrum with individual pixel weighti 
    HISTORY APSTAR:    row 2: combined spectrum with global weighting         
    HISTORY APSTAR:    row 3-nvisis+2: individual resampled visit spectra     
    HISTORY APSTAR:   unless nvists=1, which only have a single row           
    HISTORY APSTAR:  All spectra shifted to rest (vacuum) wavelength scale    
    HISTORY APSTAR:  HDU1 - Flux (10^-17 ergs/s/cm^2/Ang)                     
    HISTORY APSTAR:  HDU2 - Error (10^-17 ergs/s/cm^2/Ang)                    
    HISTORY APSTAR:  HDU3 - Flag mask (bitwise OR combined)                   
    HISTORY APSTAR:  HDU4 - Sky (10^-17 ergs/s/cm^2/Ang)                      
    HISTORY APSTAR:  HDU5 - Sky Error (10^-17 ergs/s/cm^2/Ang)                
    HISTORY APSTAR:  HDU6 - Telluric                                          
    HISTORY APSTAR:  HDU7 - Telluric Error                                    
    HISTORY APSTAR:  HDU8 - LSF coefficients                                 
    HISTORY APSTAR:  HDU9 - RV and CCF structure

    """
    # this is related to the row1
    # can also give it an oid form (band,order)
    
    # use_order = 0 
    use_order = int(use_row)-1
    hdu_header = 0  # the HDU with the header information
    hdu_flux = 1    # the HDU with the flux data
    hdu_err = 2     # the HDU with the error on the flux data
    hdu_tell = 6    # the HDU with the telluric data
    hdu_tell_er = 7 # the HDU with the error on the telluric data

    #readin_kwargs = {"non_std_fits"  :False,
    #                 "disp_type"     :'log linear',
    #                 "preferred_disp":'crval'}

    # TODO: get information from the primary header?

    def _get_obj (filename, use_order, hdu_data, hdu_error, **read_kwargs):
        meas_list,info = read_fits(filename,hdu=hdu_data,**read_kwargs)
        x_err,err_info = read_fits(filename,hdu=hdu_error)
        # err = x_err.get_data(use_order)
        # var = err**2
        # inv_var = var_2_inv_var(var)
        var = x_err[use_order].flux**2
        meas_list[use_order].inv_var = var_2_inv_var(var)
        return meas_list,info.cat(err_info,'ignore',True)
    
    data_out,data_info = _get_obj(filename,use_order,hdu_flux,hdu_err,**read_kwargs)
    if get_telluric:
        tell_out,tell_info = _get_obj(filename,use_order,hdu_tell,hdu_tell_er,**read_kwargs)
        return data_out,data_info,tell_out,tell_info
    else:
        return data_out,data_info

def read_fits_makee (filename,varience_filename=None,output_list=False,verbose=False):

    """ 
    Knows how to identify the KOA MAKEE file structure which ships with extracted data
    and apply the eyeSpec function readin to the important directories to obtain a coherent 
    spectrum object from the files


    INPUTS:
    filename : give an individual filename for the star or give the top level Star directory from MAKEE. 
               It will go from TOP_LEVEL/extracted/makee/ and use directories ccd1/ etc to find the appropriate files

    output_list : if it finds multiple chips of data it will return as a list and not a combined object


    """
    non_std_fits=False
    disp_type='default'
    preferred_disp='makee'
    

    def obj_var_2_inv_var (obj,fill=1e50):
        var = deepcopy(obj._data)

        # !! how to treat non values, i.e. negative values
        zeros = (var<=0)
        bad = (var>=fill/2.0)
        infs = (var == np.inf)

        var[zeros] = 1.0/fill
        inv_var = 1.0/var

        # set points which are very large to the fill
        inv_var[zeros] = fill
        # set points which are almost zero to zero
        inv_var[bad] = 0.0
        inv_var[infs] = 0.0

        obj._inv_var = deepcopy(inv_var)
        return inv_var


    filename = str(filename)
    if not os.path.exists(filename): raise ValueError("the given path does not exist")

    objs = {}
    inv_vars = {}

    if os.path.isdir(filename):

        if filename[-1:] != '/': filename += "/"
        
        # !! could make it smarter so it would know from anywhere within the TOP_FILE/extracted/makee/ chain
        
        full_path = filename+'extracted/makee/'
        
        if not os.path.exists(full_path): raise ValueError("Must have extracted files:"+full_path)
        
        ccds = os.listdir(full_path)
        
        for ccdir in ccds:
            if not os.path.isdir(full_path+ccdir): continue
            if not os.path.exists(full_path+ccdir+'/fits/'): continue

            if ccdir in objs.keys():
                print "Directory was already incorporated:"+ccdir
                continue

            fitspath = full_path+ccdir+'/fits/'
            fitsfiles = os.listdir(fitspath)

            print ""
            print "="*20+format("GETTING DATA FROM DIRECTORY:"+ccdir,'^40')+"="*20
            for ffname in fitsfiles:
                fname = fitspath+ffname

                # !! if file.find("_*.fits")
                # !! I could add in stuff which would go as another band for the current Flux

                if ffname.find("_Flux.fits") != -1: 
                    print "flux file:"+ccdir+'/fits/'+ffname
                    objs[ccdir] = readin(fname,preferred_disp=preferred_disp,disp_type=disp_type,non_std_fits=non_std_fits,verbose=verbose)

                elif ffname.find("_Var.fits") != -1:
                    print "variance file:"+ccdir+'/fits/'+ffname
                    tmp_obj = readin(fname,preferred_disp=preferred_disp,disp_type=disp_type,non_std_fits=non_std_fits,verbose=verbose)
                    inv_vars[ccdir] = obj_var_2_inv_var(tmp_obj)


    else:
        print "Reading in flux file:"+fname
        objs['file'] = readin(filename,preferred_disp=preferred_disp,disp_type=disp_type,non_std_fits=non_std_fits,verbose=verbose)

        if varience_filename is not None:
            inv_var = readin(varience_filename,preferred_disp=preferred_disp,disp_type=disp_type,non_std_fits=non_std_fits,verbose=verbose)
            inv_vars['file'] = obj_var_2_inv_var(inv_var)


    num_objs = 0

    OUTPUT_list = []
    OUT_header = []
    OUT_wl = []
    OUT_data = []
    OUT_inv_var = []


    for key in objs.keys():
        obj1 = objs[key]
        inv_var1 = inv_vars[key]
        # !! note I'm masking out the inf values
        mask = (inv_var1 == np.inf)
        inv_var1[mask] = 0.0

        if obj1._inv_var.shape != inv_var1.shape:
            print "HeadsUp: object and inverse variance shape are not the same"
        else: obj1._inv_var = deepcopy(inv_var1)
            
        num_objs += 1

        if output_list:
            OUTPUT_list.append(obj1)
        else:
            OUT_header.append(obj1.header)
            if obj1._wl.shape[0] > 1: print "HeadsUp: Multiple bands detected, only using the first"
            OUT_wl.append(obj1._wl[0])
            OUT_data.append(obj1._data[0])
            OUT_inv_var.append(obj1._inv_var[0])

    if output_list:
        if num_objs == 1: return OUTPUT_list[0]
        else: return OUTPUT_list        
    else:
        print ""
        print "="*30+format("COMBINING",'^20')+"="*30
        wl = np.concatenate(OUT_wl)
        data = np.concatenate(OUT_data)
        inv_var = np.concatenate(OUT_inv_var)

        obj = eyeSpec_spec(wl,data,inv_var,OUT_header[0])
        obj.hdrlist = OUT_header
        obj.filename = fitspath
        obj.edit.sort_orders()
        return obj

def read_fits_hst (filename,get_data=False):
    """
    This function is designed to read in Hubble Space Telescope Archive x1d data
    
    """
    format_error = "Unexpected HST format for fits file. Please use the X1D"
    
    try: hdulist = fits.open(filename)
    except: raise ValueError(format_error)
    
    if len(hdulist) != 2: raise ValueError(format_error)

    hdu = hdulist[1] # the data table
    
    wl = hdu.data['WAVELENGTH']
    flux = hdu.data['FLUX']
    var = hdu.data['ERROR']**2
    inv_var = var_2_inv_var(var)
    
    if get_data: return (wl,flux,inv_var)
     
    spec_obj = eyeSpec_spec(wl,flux,inv_var,hdu.header)
    
    # set up private information
    
    spec_obj.filename = filename
    spec_obj._private_info['filename'] = filename
            
    if len(hdulist) > 1: spec_obj.hdrlist = [h.header for h in hdulist]
        
    return spec_obj
 
pass
# ############################################################################# #
# output functions, perhaps part of 

def save_spectrum (spectrum,filename=None,unique_name=True,clobber=False):
    fsuffix = '.spec' #'.pkl'
    # save the spectrum object. perhaps should be part of the object itself
    pass

def save_spectrum_txt (spectrum,filename):
    """
    Outputs the eyeSpec spectrum class into a given file as text data.

    INPUTS:
    =============   ============================================================
    keyword         (type) Description
    =============   ============================================================
    spec_obj        (eyeSpec_spec) spectrum class for eyeSpec
    filename        (str) This gives the filename to save the as
    band            (int,'default') This tells which of the first dimensions of
                      spec_obj to use. 'default' is spec_obj.get_band()
    use_cropped     (bool) If True it will crop off points at the begining and 
                      end of orders which have inverse varience = 0, i.e. have
                      inf errors
    order           (int,array,None) you can specify which orders are output
                      If None then it will output all possible orders
    clobber         (bool) If True then the function will overwrite files of the
                      same file name that already exis

    include_varience (bool) If True the third column which gives the varience 
                      will be included
    divide_orders    (bool) If True it will but a commend line with '#' between
                      each of the orders
    comment          (str) What symbol to use as a comment
    divide_header    (bool,None) If False it will give one long string as the first header line
                                 If True it will divide it up by 80 character lines with a comment of '#:' 
                                 If None then no header will be printed
    =============   ============================================================

    """
    
    pass


class Information (dict):
    """
    A class for holding information about an object
    """
    
    def __init__ (self,*args,**kwargs):
        super(Information,self).__init__(*args,**kwargs)
    
    def __repr__ (self):
        reprout = 'Information {'
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
        return super(Information,self).__repr__()
    
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
        Combine two Information dictionaries together. 
        
        
        Parameters
        ----------
        other : dict subclass
            Any dictionary object will work including other Information Dictionaries
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
        info : Information 
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
            
    def guess_ra_dec(self, headers=None):
        if headers == None:
            headers = self.header_list()
        ra, dec = None, None
        for hdr in headers:
            try:
                rastr = hdr["RA"]
                decstr = hdr["DEC"]
                ra = Angle(rastr + " hour")
                dec = Angle(decstr + " degree")
                break
            except:
                pass
        return ra, dec
    
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






