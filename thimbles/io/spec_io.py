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
from collections import Iterable
import warnings

# 3rd Party
import scipy
import astropy
import astropy.io
from astropy.io import fits
import numpy as np

# Internal
from ..utils.misc import var_2_inv_var
from ..spectrum import Spectrum, WavelengthSolution
from ..metadata import MetaData


# ########################################################################### #


__all__ = ["read","read_txt","read_fits",
           "query_fits_header","WavelengthSolutionFunctions",
           "ExtractWavelengthCoefficients",
           "read","read_txt","read_fits","read_fits_hdu","read_bintablehdu",
           "read_many_fits", "read_apstar", "read_aspcap"]

# ########################################################################### #


pass
# ############################################################################ #
class query_fits_header (object):
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


def wv_soln_log_linear (pts,coefficients):

    
    
    
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
        J = np.asarray(s,dtype=int)
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







def pts_2_phys_pixels (pts,bzero=1,bscale=1):
    """
    convert points to wavelength in Angstroms via the wavelength solution
    
    Parameters
    ----------
    pts : array
        contains the points used to convert element-by-element to wavelength
    
    bzero : float
        from the header file which gives the starting point for the physical pixels
    bscale : float
        from the header file which gives the scaling for the physical pixels
        pts = bzero + pts*bscale
    
    
    bzero = query_fits_header(header,'BZERO',noval=1) # for scaled integer data, here is the zero point
    bscale = query_fits_header(header,'BSCALE',noval=0) # for scaled integer data, here is the multiplier
    
    I'm pretty sure fits uses these when it reads in the data so I don't need to
   
    """
    if bzero != 0 or bscale !=1:
        raise StandardError(("I don't know exactly what to do with bzero!=1 "
                             "or bscale!=0 :<{}><{}>".format(bzero,bscale)))
    pts = bzero + pts*bscale
    # should return [1,2,3,......,#pts]
    return pts

class WavelengthSolutionFunctions (object):
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
        if ft == 'no solution': 
            coeff = [0,1]
                
        pts,coeff,no_solution = self._check_pts_coeff(pts,coeff)        
        if no_solution: 
            ft = 'no solution'
        
        if ft not in self._function_types:
            if default_function: 
                ft = 'no solution'
            else:
                raise ValueError("Unknown function type:"+str(function_type))
        
        if ft is 'pts': 
            return self.wl_soln_pts(pts)
        
        func = self._function_types[ft][1]
        return func(pts,coeff)

    
wavelength_solution_functions = WavelengthSolutionFunctions()

class WavelengthSolutionCoefficients (object):
    """
    This class holds the coefficients as extracted from the fits header
    
    """
    def __init__ (self, equ_type='none' , extra='none'):
        """
        equ_type correpsonds to the WavelengthSolutionFunctions
        coefficients correponds to the function but for a polynomial
             coefficients = c
             y = c[0]+c[1]*x+c[2]*x**2+c[3]*c**3+....
        extra     extra info from the extraction process
        """
        self.equ_type = self.set_equation_type(equ_type)
        self.coefficients = []
        self.extra = extra
        self.rv = 0.0
    
    def __len__ (self):
        return len(self.coefficients)
    
    def add_coeffs (self,coeff,index=None):
        if index is None: 
            self.coefficients.append(coeff)
        else: 
            self.coeff[index] = coeff
        
    def get_coeffs (self,index=None):
        if index not in xrange(len(self.coefficients)): 
            index = len(self.coefficients)-1
        return self.coefficients[index]
    
    def get_equation_type (self):
        return deepcopy(self.equ_type)
    
    def set_equation_type (self,equ_type):
        self.equ_type = wavelength_solution_functions.get_func_name(equ_type)

class ExtractWavelengthCoefficients (object):
    """
    An object for extracting wavelengths from a fits header

    The way that a wavelength solution (function from pixel space to 
    wavelengths) has many different formats. This attempts to use the header to
    check many formats and extract coefficients and function type for the 
    wavelength solution. 
    
    Ambiguities exist when several formats are specified. These are resolved in 
    the order set by class attribute resolve_wvsoln
    
    Parameters
    ----------
    fits_header : `astropy.io.fits.header.Header`
    

    
    
    """

    resolve_wvsoln = ['spectre',
                      'wv_0',
                      # 'co_0',
                      'w0',
                      'wcs',
                      'crvl',
                      'ctype1',
                      'linear',
                      'no solution']
    
    def __init__ (self,fits_header):
        self.header = fits_header
        
        results = [self.from_SPECTRE,
                   self.from_makee_wv,
                   self.from_w0,
                   self.from_wcs,
                   self.from_crvl,
                   self.from_ctype1,
                   self.from_pts,
                   self.from_none]
        
        # This is the order to resolve the coefficients in
        self.wvsoln_func = {}
        for key,value in zip(self.resolve_wvsoln,results):
            self.wvsoln_func[key] = value
    
    def check_preferred (self,preferred):
        """
        Checks if preferred is in the list of preferred values
        
        Parameters
        ----------
        preferred : None or {0}
        
        Raises
        ------
        ValueError : if preferred is not an acceptable value
        
        """.format(", ".format(self.resolve_wvsoln))
        if preferred is None:
            return None
        if preferred.lower() in self.resolve_wvsoln:
            return preferred.lower()
        else:
            raise ValueError("preferred not in "+", ".join(self.resolve_wvsoln))             
        
    def __repr__ (self):
        output = "ExtractingWavelengthCoefficients\n"
        output += repr(self.header)
        return output

    def __iter__ (self):
        return iter(self.wvsoln_func.iteritems())
      
    def __getitem__ (self,coeff_type):
        if coeff_type not in self.wvsoln_func:
            raise KeyError("Unknown coeff type")
        return self.wvsoln_func[coeff_type]()
      
    def get_coeffs (self,preferred=None,all_=False):
        # return all wavelength solution objects
        if all_:
            return {key:wvfunc() for key,wvfunc in self.wvsoln_func}
        
        # check and return preferred wavelength solution
        pref = self.check_preferred(preferred)
        if pref is not None:
            return self[pref]
        
        # resolve wavelength solution
        for wvsoln_name in self.resolve_wvsoln:
            wvcoeff = self.wvsoln_func[wvsoln_name]()
            if not len(wvcoeff):
                continue
            return wvcoeff
    
    pass
    # --------------------------------------------------------------------------- #
    
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
      
        def get_order_wvsoln (ordi):
            if ordi not in order_coeff: 
                if i == 0: return False
                return True
            # w = sum from i=1 to nfunc {wt_i * (w0_i + W_i(p)) / (1 + z)}
            if order_coeff[ordi][2] != dcflag: 
                raise ValueError("Encountered orders with different functions dcflag1="+str(dcflag)+"  dcflag"+str(ordi)+"="+str(order_coeff[ordi][2]))
            z = order_coeff[ordi][6]
            if z != 0: 
                wlcoeff.rv = 1.0/(10.+z)
        
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
            if get_order_wvsoln(i): break
            
        wlcoeff.extra = ['used header to get parameters and coefficients, function: '+equ_type+', to apply wl = function(pts)',order_coeff]
        return wlcoeff
    
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

    def from_pts (self):
        wlcoeff = WavelengthSolutionCoefficients()
        wlcoeff.extra = 'no header info used, just a wl = pts'
        wlcoeff.set_equation_type('linear')
        wlcoeff.add_coeffs([0,1])        
        return wlcoeff

    def from_none (self):
        wlcoeff = WavelengthSolutionCoefficients()
        wlcoeff.extra = 'following checks found that the first order coefficient is zero, setting basic linear wl = pts'
        wlcoeff.set_equation_type('linear')
        wlcoeff.add_coeffs([0,1])        
        return wlcoeff        
    
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
    
    def from_SPECTRE (self):
        """
        Extracts the wavelength solution from the SPECTRE HISTORY tags
        """
        wlcoeff = WavelengthSolutionCoefficients()
        if not query_fits_header(self.header,'HISTORY',noval=0).found: 
            return wlcoeff # old SPECTRE-stype dispersion information

        #==========================================================================#
        spectre_history = self.get_SPECTRE_history()   
        # if you found history lines use one with the most recent data                
        if len(spectre_history) > 0:
            most_recent = sorted(spectre_history.keys())[-1]
                
            extra_data, disp_type, coeff = spectre_history[most_recent]
            
            wlcoeff.extra = extra_data
            wlcoeff.set_equation_type(disp_type)
            wlcoeff.add_coeffs(coeff)
            
        # NOTE: spectre_history has all the history tags for spectre keyed by time stamp                          
        return wlcoeff
        
    def get_SPECTRE_history (self):
    
        histories = self.header['HISTORY']   
        get_spectre_d = lambda x: x[11:17]
        
        def parse_timetag (hist_line):
            date_str = hist_line[:10]
            day,month,year = [int(s) for s in date_str.split(":")]
            timetag = time.mktime((year,month,day,0,0,0,0,0,0))
            return timetag
            
        def parse_coefficients (hist_line):
            coeffs = hist_line[18:36],hist_line[36:54],hist_line[54:]
            coeffs = [float(c.replace("D","e")) for c in coeffs]
            return coeffs
            
        spectre_history = {}    
        for i,hist_line in enumerate(histories):
            ds1 = get_spectre_d(hist_line) 
            if ds1 != 'D1,2,3':
                continue
            
            if i+2 >= len(histories):
                continue
            
            # get the current and next two history lines
            hl1 = hist_line
            hl2 = histories[i+1]
            hl3 = histories[i+2]
            
            # check tags
            ds2 = get_spectre_d(hl2)
            ds3 = get_spectre_d(hl3)
            if ds2 != "D4,5,6" or ds3 != "D7,8,9":
                warnings.warn("Expected next two history lines "
                              "to have D4,5,6 and D7,8,9")
                continue
                
            # check time stamp
            tt1 = parse_timetag(hl1)
            tt2 = parse_timetag(hl2)
            tt3 = parse_timetag(hl3)
            if not (tt1==tt2 and tt2==tt3):
                # time for these tags must be the same
                continue 
        
            c1 = parse_coefficients(hl1)
            c2 = parse_coefficients(hl2)
            disp_info = parse_coefficients(hl3)
            coeff = c1+c2 
            
            # cheby poly may need disp_info[0]
            # disp_info[0] == c(7)
            # disp_info[1] == c(8)
            # from SPECTRE: 
            # c20    p = (point - c(6))/c(7)
            # c      xpt = (2.*p-(c(9)+c(8)))/(c(9)-c(8))
            if disp_info[2] == 1: 
                disp_type = 'chebyshev poly'
            elif disp_info[2] == 2: 
                disp_type = 'legrendre poly'
            elif disp_info[2] == 3:
                warnings.warn("check the output, I think I may need to use disp_info[1]  (timbles.io.io.ExtractWavelengthCoefficients) ")
                # from SPECTRE: s = (point-1.)/(real(npt)-1.)*c(8)
                # c(8) == disp_info[1] ==> true
                disp_type = 'spline3'
            else: 
                disp_type = 'poly' # but likely the orders > 1 have coefficients zero
       
            extra_data= ['used header to get SPECRE HISTORY tags, function:'+disp_type+', to apply wl=function(pts)',[hl1,hl2,hl3]]
            spectre_history[tt1] = (extra_data,disp_type,coeff) 
                
        return spectre_history
        
        spectre_history = {}    
        for i,hist_line in enumerate(histories):
            ds1 = get_spectre_d(hist_line) 
            if ds1 != 'D1,2,3':
                continue
            
            if i+2 >= len(histories):
                continue
            
            # get the current and next two history lines
            hl1 = hist_line
            hl2 = histories[i+1]
            hl3 = histories[i+2]
            
            # check tags
            ds2 = get_spectre_d(hl2)
            ds3 = get_spectre_d(hl3)
            if ds2 != "D4,5,6" or ds3 != "D7,8,9":
                warnings.warn("Expected next two history lines "
                              "to have D4,5,6 and D7,8,9")
                continue
                
            # check time stamp
            tt1 = parse_timetag(hl1)
            tt2 = parse_timetag(hl2)
            tt3 = parse_timetag(hl3)
            if not (tt1==tt2 and tt2==tt3):
                # time for these tags must be the same
                continue 
        
            c1 = parse_coefficients(hl1)
            c2 = parse_coefficients(hl2)
            disp_info = parse_coefficients(hl3)
            coeff = c1+c2 
            
            # cheby poly may need disp_info[0]
            # disp_info[0] == c(7)
            # disp_info[1] == c(8)
            # from SPECTRE: 
            # c20    p = (point - c(6))/c(7)
            # c      xpt = (2.*p-(c(9)+c(8)))/(c(9)-c(8))
            if disp_info[2] == 1: 
                disp_type = 'chebyshev poly'
            elif disp_info[2] == 2: 
                disp_type = 'legrendre poly'
            elif disp_info[2] == 3:
                warnings.warn("check the output, I think I may need to use disp_info[1]  (timbles.io.io.ExtractWavelengthCoefficients) ")
                # from SPECTRE: s = (point-1.)/(real(npt)-1.)*c(8)
                # c(8) == disp_info[1] ==> true
                disp_type = 'spline3'
            else: 
                disp_type = 'poly' # but likely the orders > 1 have coefficients zero
       
            extra_data= ['used header to get SPECRE HISTORY tags, function:'+disp_type+', to apply wl=function(pts)',[hl1,hl2,hl3]]
            spectre_history[tt1] = (extra_data,disp_type,coeff) 
                
        return spectre_history
        
pass
# ############################################################################# #

# read functions

def read_txt (filepath,**np_kwargs):
    """
    Readin text files with wavelength and data columns (optionally inverse varience)
    
    Parameters
    ----------
    filepath : string
        Gives the path to the text file
    np_kwargs : dictionary
        Contains keywords and values to pass to np.loadtxt
        This includes things such as skiprows, usecols, etc.
        unpack and dtype are set to True and float respectively 
    
    Returns
    -------
    spectrum : list of `thimbles.spectrum.Spectrum` objects 
        If get_data is False returns a Spectrum object
    
    Notes
    -----
    __1)__ Keywords txt_data, unpack, and dtype are forced for the
        np_kwargs.
        
    """ 
    #### check if file exists   ####### #############
    if not os.path.isfile(filepath): 
        raise IOError("File does not exist:'{}'".format(filepath))

    metadata = MetaData()
    metadata['filepath'] = os.path.abspath(filepath)

    # Allows for not repeating a loadtxt
    np_kwargs['unpack'] = True
    np_kwargs['dtype'] = float
    txt_data = np.loadtxt(filepath,**np_kwargs)
    
    # check the input txt_data
    if txt_data.ndim == 1:
        warnings.warn("NO WAVELENGTH DATA FOUND, USING FIRST COLUMN AS DATA")
        data = txt_data 
        wvs = np.arange(len(data))+1
        var = None
    elif txt_data.ndim == 2:
        wvs = txt_data[0]
        data = txt_data[1]
        var = None
    elif txt_data.ndim == 3: 
        wvs,data,var = txt_data
    elif txt_data.shape[0] > 2: 
        warnings.warn(("Found more than 3 columns in text file '{}' "
                       "taking the first three to be wavelength, data,"
                       " variance respectively".format(filepath)))
        wvs,data,var = txt_data[:3]
        
    if var is not None:        
        inv_var = var_2_inv_var(var)
    else:
        inv_var = None
    
    return [Spectrum(wvs,data,inv_var,metadata=metadata)]
    
############################################################################
# readin is the main function for input

def read_fits (filepath, which_hdu=0, band=0, preferred_wvsoln=None):
    if not os.path.isfile(filepath): 
        raise IOError("File does not exist:'{}'".format(filepath))
        # TODO: add more checks of the fits header_line to see
        # if you can parse more about it. e.g. it's an apogee/hst/or/makee file     
    hdulist = fits.open(filepath)
    if len(hdulist) > 1 and isinstance(hdulist[1],astropy.io.fits.hdu.table.BinTableHDU):
        return read_bintablehdu(hdulist)
    
    kws = dict(which_hdu=which_hdu,
               band=band,
               preferred_wvsoln=preferred_wvsoln)
    return read_fits_hdu(hdulist,**kws)

read_fits.__doc__ = """
    Takes a astropy.io.fits hdulist and then for a particular hdu and band
    extracts the wavelength and flux information
    
    This goes through keywords in the header and looks for specific known
    keywords which give coefficients for a wavelenth solution. It then 
    calculates the wavelengths based on that wavelength solution.
    
    Parameters
    ----------
    hdulist : `astropy.io.fits.HDUList` or string
        A header unit list which contains all the header units
    which_hdu : integer
        Which hdu from the hdulist to use, start counting at 0
    band : integer
        If the hdu has NAXIS3 != 0 then this will select which
        value that dimension should be
    preferred_wvsoln : None or "{}"
        
    Returns
    -------
    spectra : list of `thimbles.spectrum.Spectrum` objects 
    
    Raises
    ------
    IOError : If it encounters unknown KEYWORD options when looking
        for a wavelength solution
    
    """.format(", ".format(ExtractWavelengthCoefficients.resolve_wvsoln))

def read_fits_hdu (hdulist,which_hdu=0,band=0,preferred_wvsoln=None):
    """
    Reads a fits header unit which contains a wavelength solution
    """
    hdu = hdulist[which_hdu]
    header = hdu.header
    if query_fits_header(header,'APFORMAT').found: 
        warnings.warn(("Received keyword APFORMAT,"
                       " no machinary to deal with this."
                       " [thimbles.io.read_fits_hdu]"))
    # bzero = query_fits_header(header,"BZERO",noval=0)
    # bscale = query_fits_header(header,"BSCALE",noval=1)
    
    ###### read in data ##############################################
    data = hdu.data
    
    # if there's no data return empty
    if data is None:
        warnings.warn("no data for header unit [thimbles.io.read_fits_hdu]")
        wvs, flux, inv_var = np.zeros((3,1))
        return [Spectrum(wvs,flux,inv_var)]
    
    # hdu selection, band select
    if data.ndim == 3:
        data = data[band]
    elif data.ndim == 1:
        data = data.reshape((1,-1))
    # now the data is ndim==2 with the first dimension as the orders
    # and the second being the data points
    
    ##### Calculate the wavelengths for the data
    # set up wavelength and inverse_variance
    wvs = np.ones(data.shape)
    
    # get the wavelength coefficients
    extract_wvcoeffs = ExtractWavelengthCoefficients(hdu.header)
    wvcoeff = extract_wvcoeffs.get_coeffs(preferred_wvsoln)
    
    idx_orders = xrange(len(data))
    
    # go through all the orders
    do_progress = True
    progressive_pt = 1 # this will advance and be used when there is no wavelength solution
    
    for i in idx_orders:
        # get the coefficients and function type    
        equ_type = wvcoeff.get_equation_type()
        if equ_type in ('none',None,'no solution') and do_progress: 
            coeff = [progressive_pt,1]
            equ_type = 'pts'
        else: 
            coeff = wvcoeff.get_coeffs(i)
        # pts[0] = 1 :: this was definitely the right thing to do for SPECTRE's 1-D output but may not be for other equations, may need pts[0]=0,  this may be for bzero,bscale
        pts = np.arange(len(wvs[i]))+1
        # apply function
        wvs[i] = wavelength_solution_functions(pts, coeff, equ_type)    
        progressive_pt += len(pts)

    #=================================================================#
    metadata = MetaData()
    metadata['filepath'] = hdulist.filename
    metadata['hdu_used']=which_hdu
    metadata['band_used']=band
    metadata['wavelength_coeffs'] = wvcoeff
    metadata['header0'] = deepcopy(hdu.header)
        
    spectra = []
    for i in idx_orders:
        metadata['order'] = i
        wl_soln = WavelengthSolution(wvs[i],rv=wvcoeff.rv)
        spectra.append(Spectrum(wl_soln,data[i],metadata=metadata))
         
    return spectra

def read_bintablehdu (hdulist,which_hdu=1,wvcolname=None,fluxcolname=None,varcolname=None):
    """
    Read in a fits binary fits table
    
    Parameters 
    ----------
    bintablehdu : `astropy.io.fits.BinTableHDU`
    wvcolname : None or string 
    fluxcolname : None or string
    varcolname : None or string
    
    
    """
    metadata = MetaData()
    metadata['filepath'] = hdulist.filename()
    for i,hdu in enumerate(hdulist):
        metadata['header{}'.format(i)] = hdu.header
    
    guesses = dict(wvcol_guess = ['wavelength','wvs','wavelengths'],
                   fluxcol_guess = ['flux','ergs','intensity'],
                   var_guess = ['variance','varience'],
                   inv_var_guess = ['inv_var','inverse variance','inverse varience'],
                   sigma_guess = ['sigma','error'],
                   )   
    # include all uppercase and capitalized guesses too
    items = guesses.items()     
    for key,values in items:
        guesses[key] += [v.upper() for v in values]
        guesses[key] += [v.capitalize() for v in values]
    
    def get_key (set_key,options,guess_keys):
        if set_key is not None:
            return set_key
        else:
            for key in guess_keys:
                if key in options:
                    return key
    
    which_hdu = abs(which_hdu)
    
    if not (len(hdulist) > which_hdu and isinstance(hdulist[which_hdu],astropy.io.fits.hdu.table.BinTableHDU)):
        raise ValueError("This must be done with a bin table fits file")
        
    # read in data   
    data = hdulist[which_hdu].data
    options = data.dtype.names
    
    # get the keys
    wvs_key = get_key(wvcolname,options,guesses['wvcol_guess'])
    flux_key = get_key(fluxcolname,options,guesses['fluxcol_guess'])
    
    sig_key = get_key(None,options,guesses['sigma_guess'])
    var_key = get_key(varcolname,options,guesses['var_guess'])
    inv_var_key = get_key(None,options,guesses['inv_var_guess'])
    
    # check that these keys are essential
    if wvs_key is None or flux_key is None:
        raise ValueError("No keys which make sense for wavelengths or flux")
    wvs = data[wvs_key]
    flux = data[flux_key]
    
    # check for error keys
    if var_key is not None:
        inv_var = var_2_inv_var(data[var_key])
    elif sig_key is not None:
        var = data[sig_key]**2
        inv_var = var_2_inv_var(var)
    elif inv_var_key is not None:
        inv_var = data[inv_var_key]
    else:
        inv_var = None
    
    # store the spectra 
    spectra = []
    if wvs.ndim == 2:
        for i in xrange(len(wvs)):
            if inv_var is not None:
                ivar = inv_var[i]
            else:
                ivar = None
            metadata['order'] = i
            spectra.append(Spectrum(wvs[i],flux[i],ivar,metadata=metadata.copy()))
    elif wvs.ndim == 1:
        spectra.append(Spectrum(wvs,flux,inv_var,metadata=metadata))
    else:
        raise ValueError("Don't know how to deal with data ndim={}".format(wvs.ndim))
    return spectra

def read_many_fits (filelist,relative_paths=False,are_orders=False):
    """
    takes a list of spectre 1D files and returns `timbles.Spectrum` objects 

    Parameters
    ----------
    filelist : string or list of strings
        Each string gives the file path to a 
    relative_paths : boolean
        If True then each file path will be treated relative to the filelist file 
        directory.
    are_orders : boolean
        If True it will order files by wavelength and assign them order numbers

    Return
    ------
    spectra : list of `thimbles.Spectrum` objects
    
    
    """
    list_of_files = []
    relative_paths = bool(relative_paths)
    nofile = "File does not exist '{}'"

    if isinstance(filelist,(basestring)):
        dirpath = os.path.dirname(filelist)
        
        if not os.path.isfile(filelist):
            raise IOError(nofile.format(filelist))
        
        with open(filelist) as f:
            files = f.readlines()
            
        for fname in files:
            fname = fname.split()[0]
            if relative_paths:
                fname = os.path.join(dirpath,fname)  
            if not os.path.isfile(fname):
                warnings.warn(nofile.format(filelist))                      
            else: 
                list_of_files.append(fname)
        f.close()
    #-----------------------------------------------#
    # if given input is not a string assume it's a list/array
    elif isinstance(filelist,Iterable):
        list_of_files = [str(fname).split()[0] for fname in filelist]
    else:
        raise TypeError("filelist must be string or list of strings")
    
    #============================================================#
    
    spectra = []
    for fname in list_of_files:
        spectra += read(fname)
        
    if are_orders:
        med_wvs = []
        for spec in spectra:
            med_wvs.append(np.median(spec.wv))
        sort_spectra = []
        for i in np.argsort(med_wvs):
            spec = spectra[i]
            spec.metadata['order'] = i
            sort_spectra.append(spec)
        spectra = sort_spectra     
    return spectra

pass
# ############################################################################# #
# This is the general read in function for fits and text files

# general read function

def is_many_fits (filepath):
    
    if isinstance(filepath,basestring):
        # check if this is a file with a list of files
        if not os.path.isfile(filepath):
            return False
        with open(filepath,'r') as f:
            for line in f:
                if len(line.strip()) and line.strip()[0] != "#" and not os.path.isfile(line.rstrip().split()[0]):
                    return False
        return True    
    elif isinstance(filepath,Iterable):
        return True
    else:
        return False

def read (filepath,**kwargs):
    """
    General read 
    
    **kwargs are passed either to read_txt or read_fits depending on determined 
    file type
    
    """
    # NOTE: Could also check for 'END' card and/or 'NAXIS  ='
    fits_rexp = re.compile("[SIMPLE  =,XTENSION=]"+"."*71+"BITPIX  =")
    
    # check the first line of the file. What type is it?
    with open(filepath,'r') as f:
        header_line = f.readline()
    s = fits_rexp.search(header_line)
    
    
    if s is None: # is not a fits file
        if is_many_fits(filepath): # is a file of many files
            return read_many_fits(filepath,**kwargs)
        else: # is a single text file
            return read_txt(filepath,**kwargs)
    else: # is a fits file
        return read_fits(filepath,**kwargs)

pass
# ############################################################################# #
# special readin functions

def read_apstar (filepath, data_hdu=1, error_hdu=2, row=0):
    """ 
  reads the apStar APOGEE fits files.
    
    Paremeters
     ----------
     filepath : string path to fits file
         APOGEE pipeline reduced data with a 0 header unit similar to the below
     use_row : integer 
         APOGEE refers to these as rows, default is row1 ("combined spectrum with individual pixel weighting")
     get_telluric : boolean
         If True then it will also extract the telluric data
 
 
    Returns
     -------
    a list with a single apogee spectrum in it
     
 
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
    hdulist = fits.open(filepath)
    metadata = MetaData()
    metadata['filepath'] = hdulist.filename()
    hdr = hdulist[0].header
    metadata['header'] = hdr
    
    if len(hdulist[1].data.shape) == 2:
        flux = hdulist[data_hdu].data[row].copy()
        sigma = hdulist[error_hdu].data[row].copy()
    elif len(hdulist[1].data.shape) == 1:
        flux = hdulist[data_hdu].data
        sigma = hdulist[error_hdu].data
    crval1 = hdr["CRVAL1"]
    cdelt1 = hdr["CDELT1"]
    nwave  = hdr["NWAVE"]
    wv = np.power(10.0, np.arange(nwave)*cdelt1+crval1)
    return [Spectrum(wv, flux, var_2_inv_var(sigma**2))]


def read_aspcap(filepath):
    hdulist = fits.open(filepath)
    metadata = MetaData()
    metadata['filepath'] = hdulist.filename()
    hdr = hdulist[0].header
    metadata['header'] = hdr
    
    flux = hdulist[1].data
    sigma = hdulist[2].data
    invvar = var_2_inv_var(sigma**2)*(flux > 0)
    crval1 = hdulist[1].header["CRVAL1"]
    cdelt1 = hdulist[1].header["CDELT1"]
    nwave  = len(flux)
    wv = np.power(10.0, np.arange(nwave)*cdelt1+crval1)
    return [Spectrum(wv, flux, invvar)]

# 
# def read_fits_makee (filepath,varience_filepath=None,output_list=False,verbose=False):
# 
#     """ 
#     Knows how to identify the KOA MAKEE file structure which ships with extracted data
#     and apply the eyeSpec function readin to the important directories to obtain a coherent 
#     spectrum object from the files
# 
# 
#     INPUTS:
#     filepath : give an individual filepath for the star or give the top level Star directory from MAKEE. 
#                It will go from TOP_LEVEL/extracted/makee/ and use directories ccd1/ etc to find the appropriate files
# 
#     output_list : if it finds multiple chips of data it will return as a list and not a combined object
# 
# 
#     """
#     non_std_fits=False
#     disp_type='default'
#     preferred_disp='makee'
#     
# 
#     def obj_var_2_inv_var (obj,fill=1e50):
#         var = deepcopy(obj._data)
# 
#         # !! how to treat non values, i.e. negative values
#         zeros = (var<=0)
#         bad = (var>=fill/2.0)
#         infs = (var == np.inf)
# 
#         var[zeros] = 1.0/fill
#         inv_var = 1.0/var
# 
#         # set points which are very large to the fill
#         inv_var[zeros] = fill
#         # set points which are almost zero to zero
#         inv_var[bad] = 0.0
#         inv_var[infs] = 0.0
# 
#         obj._inv_var = deepcopy(inv_var)
#         return inv_var
# 
# 
#     filepath = str(filepath)
#     if not os.path.exists(filepath): raise ValueError("the given path does not exist")
# 
#     objs = {}
#     inv_vars = {}
# 
#     if os.path.isdir(filepath):
# 
#         if filepath[-1:] != '/': filepath += "/"
#         
#         # !! could make it smarter so it would know from anywhere within the TOP_FILE/extracted/makee/ chain
#         
#         full_path = filepath+'extracted/makee/'
#         
#         if not os.path.exists(full_path): raise ValueError("Must have extracted files:"+full_path)
#         
#         ccds = os.listdir(full_path)
#         
#         for ccdir in ccds:
#             if not os.path.isdir(full_path+ccdir): continue
#             if not os.path.exists(full_path+ccdir+'/fits/'): continue
# 
#             if ccdir in objs.keys():
#                 print "Directory was already incorporated:"+ccdir
#                 continue
# 
#             fitspath = full_path+ccdir+'/fits/'
#             fitsfiles = os.listdir(fitspath)
# 
#             print ""
#             print "="*20+format("GETTING DATA FROM DIRECTORY:"+ccdir,'^40')+"="*20
#             for ffname in fitsfiles:
#                 fname = fitspath+ffname
# 
#                 # !! if file.find("_*.fits")
#                 # !! I could add in stuff which would go as another band for the current Flux
# 
#                 if ffname.find("_Flux.fits") != -1: 
#                     print "flux file:"+ccdir+'/fits/'+ffname
#                     objs[ccdir] = read_fits(fname,preferred_disp=preferred_disp,disp_type=disp_type,non_std_fits=non_std_fits,verbose=verbose)
# 
#                 elif ffname.find("_Var.fits") != -1:
#                     print "variance file:"+ccdir+'/fits/'+ffname
#                     tmp_obj = read_fits(fname,preferred_disp=preferred_disp,disp_type=disp_type,non_std_fits=non_std_fits,verbose=verbose)
#                     inv_vars[ccdir] = obj_var_2_inv_var(tmp_obj)
# 
# 
#     else:
#         print "Reading in flux file:"+fname
#         objs['file'] = read_fits(filepath,preferred_disp=preferred_disp,disp_type=disp_type,non_std_fits=non_std_fits,verbose=verbose)
# 
#         if varience_filepath is not None:
#             inv_var = read_fits(varience_filepath,preferred_disp=preferred_disp,disp_type=disp_type,non_std_fits=non_std_fits,verbose=verbose)
#             inv_vars['file'] = obj_var_2_inv_var(inv_var)
# 
# 
#     num_objs = 0
# 
#     OUTPUT_list = []
#     OUT_header = []
#     OUT_wl = []
#     OUT_data = []
#     OUT_inv_var = []
# 
# 
#     for key in objs.keys():
#         obj1 = objs[key]
#         inv_var1 = inv_vars[key]
#         # !! note I'm masking out the inf values
#         mask = (inv_var1 == np.inf)
#         inv_var1[mask] = 0.0
# 
#         if obj1._inv_var.shape != inv_var1.shape:
#             print "HeadsUp: object and inverse variance shape are not the same"
#         else: obj1._inv_var = deepcopy(inv_var1)
#             
#         num_objs += 1
# 
#         if output_list:
#             OUTPUT_list.append(obj1)
#         else:
#             OUT_header.append(obj1.header)
#             if obj1._wl.shape[0] > 1: print "HeadsUp: Multiple bands detected, only using the first"
#             OUT_wl.append(obj1._wl[0])
#             OUT_data.append(obj1._data[0])
#             OUT_inv_var.append(obj1._inv_var[0])
# 
#     if output_list:
#         if num_objs == 1: return OUTPUT_list[0]
#         else: return OUTPUT_list        
#     else:
#         print ""
#         print "="*30+format("COMBINING",'^20')+"="*30
#         wl = np.concatenate(OUT_wl)
#         data = np.concatenate(OUT_data)
#         inv_var = np.concatenate(OUT_inv_var)
# 
#         obj = eyeSpec_spec(wl,data,inv_var,OUT_header[0])
#         obj.hdrlist = OUT_header
#         obj.filepath = fitspath
#         obj.edit.sort_orders()
#         return obj
# 
# def save_spectrum_txt (spectrum,filepath):
#     """
#     Outputs the eyeSpec spectrum class into a given file as text data.
# 
#     INPUTS:
#     =============   ============================================================
#     keyword         (type) Description
#     =============   ============================================================
#     spec_obj        (eyeSpec_spec) spectrum class for eyeSpec
#     filepath        (str) This gives the filename to save the as
#     band            (int,'default') This tells which of the first dimensions of
#                       spec_obj to use. 'default' is spec_obj.get_band()
#     use_cropped     (bool) If True it will crop off points at the begining and 
#                       end of orders which have inverse varience = 0, i.e. have
#                       inf errors
#     order           (int,array,None) you can specify which orders are output
#                       If None then it will output all possible orders
#     clobber         (bool) If True then the function will overwrite files of the
#                       same file name that already exis
# 
#     include_varience (bool) If True the third column which gives the varience 
#                       will be included
#     divide_orders    (bool) If True it will but a commend line with '#' between
#                       each of the orders
#     comment          (str) What symbol to use as a comment
#     divide_header    (bool,None) If False it will give one long string as the first header line
#                                  If True it will divide it up by 80 character lines with a comment of '#:' 
#                                  If None then no header will be printed
#     =============   ============================================================
# 
#     """
#     
#     pass






