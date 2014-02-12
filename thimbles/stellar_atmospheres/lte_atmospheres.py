# This is used for creating kurucz atmospheres

#=============================================================================#
# import modules
import numpy as np
import os
import subprocess
from .utils import get_filename,get_model_name, Batom
from . import executables 

pass
#=============================================================================#
def check_makekurucz (error=False):
    _makekurucz_avail_error = "Must have makekurucz executable declared to use this function not: "+executables['makekurucz']
    check =  os.path.isfile(executables['makekurucz'])
    if not check and error: raise ValueError(_makekurucz_avail_error)
    return check

def check_marcs (error=False):
    _avail_error = "Must have the marcs model executable declared to use this function not: "+executables['marcs']
    check =  os.path.isfile(executables['marcs'])
    if not check and error: raise ValueError(_avail_error)
    return check
        
pass
#=============================================================================#
# functions for creating the kurucz atmospheres


def create_kurucz_atmo_model (teff,logg,feh,turb,modtype='ODFNEW',filename='FINALMODEL',verbose=True, clean_up=True):
    """
    To use the makekurucz Fortan interpreter to create stellar atmosphere models for MOOG
        
    Parameters
    ----------
    teff : float 
        Stellar effective temperature
    logg : float 
        Gravitational acceleration at the surface of the star
    feh  : float 
        Normalized solar metallicity [Fe/H]
    vt   : float 
        Stellar microturbulence
    modtype : string 
        the type of model used, Possible Choices:
        * ODFNEW -> 
        * AODFNEW -> Same as the ODFNEW only with an alpha enhancement
        * NOVER ->
        * KUROLD ->
    filename : string 
        Give the output filename. If 'rename' it will use the naming convention
        {teff}p{logg}mp{feh}p{vt}.modtype
    verbose : boolean 
        If 'True' you get verbose output
    clean_up : boolean 
        Will remove files : M1, M2, MOD*
        
    Returns
    -------
    model_created : boolean 
        Model creation, if the parameters were correct and the model created it returns True
    
    
    Raises
    ------
    None
    
    
    Notes
    -----
    __1)__ There are limits in the interpolation of Kurucz. If a limit is passed the function will return False
        3500 < teff < 10000
        0 < logg < 5
        -5 < feh < 1.0

    __2)__ If you give a feh > 1.0 then it will multiply by negative 1 (e.g. 2.13==> -2.13)

    __3)__ The individual modtypes may have inherent limits as well (e.g. feh in ODFNEW >= -2.5)
    
    __4)__ This function won't work if the executables['makekurucz'] is not declared
    
    __5)__ This will also overwrite any file names 'FINALMODEL'
    
    
    
    Examples
    --------
    >>> create_kurucz_atmo_model(5000, 4.1, -2.13, 1.1, modtype='aodfnew', filename='rename')
    >>>
    >>> ls
    5000p410m213p110.aodfnew

    """
    check_makekurucz(True)
    MAKEKURUCZ_EXE = executables['makekurucz']
    
    # impose kurucz limits
    if not (3500 < teff < 10000):
        if verbose: 
            print "Teff is out of kurucz range (3500,10000): "+str(teff)
        return False

    if not (0 < logg < 5):
        if verbose: 
            print "Logg is out of kurucz range (0,5): "+str(logg)
        return False

    if feh > 1.0: 
        if verbose: 
            print "[Fe/H] given larger than 1, multiplying by negative one"
        feh *= -1.0
        
    if not (-5 < feh < 1.0):
        if verbose: 
            print "[Fe/H] is out of kurucz range (-5,1): "+str(feh)
        return False

    modtype=modtype.upper()
    pos_modtypes = ['KUROLD', 'ODFNEW', 'AODFNEW', 'NOVER']
    if modtype not in pos_modtypes:
        if verbose: 
            print "Model type must be in: "+", ".join(pos_modtypes)
        return False

    inp = " ".join([str(teff), str(logg), str(feh), str(turb),"\n"+modtype])+"\n"

    devnull = open('/dev/null', 'w')
    makeKurucz = subprocess.Popen(MAKEKURUCZ_EXE, stdin=subprocess.PIPE,stdout=devnull)
    devnull.close()
    makeKurucz.communicate(input=inp)

    if filename == 'rename': 
        filename = get_model_name(teff,logg,feh,turb,modtype)
    if filename != 'FINALMODEL': 
        os.system("mv FINALMODEL "+str(filename))
    if clean_up: 
        _ = subprocess.Popen(["rm","-f","MOD*","M1","M2"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        
    return True

def create_marcs_atmo_model (teff, logg, feh, turb, mass, grid_vturb, grid_alpha, filename = "FINALMODEL"):
    check_marcs(True)
    MARCS_EXE = executables['marcs']
    os.system(MARCS_EXE+" %5.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %s\n" % (teff, logg, feh, turb, mass, grid_vturb, grid_alpha, filename))
    return True

def create_lte_atmosphere (model_pars, modtype='ODFNEW',filename='FINALMODEL',verbose=True,clean_up=True):
    model_pars = list(model_pars)
    if modtype == 'marcs': return create_marcs_atmo_model(*model_pars[:6],filename=filename)
    else: return create_kurucz_atmo_model(*model_pars[:4],modtype=modtype,filename=filename,verbose=verbose,clean_up=clean_up)

def run_create_lte_atmosphere ():
    """
PURPOSE:
   This interactively runs create_kurucz_atmo_model and prompts the user for the inputs
   
CATEGORY:
   Stellar Atmospheres

DEPENDENCIES:
   External Modules Required
   =================================================
    os, numpy
   
   External Functions and Classes Required
   =================================================
    create_kurucz_atmo_model
    
NOTES:
    
EXAMPLE:
   >>> run_create_atmo_model()

MODIFICATION HISTORY:
    13, Jun 2013: Dylan Gregersen

    """  
    check_makekurucz(True)
    def convert_modtype (modtype):
        if modtype.lower()[0] == 'o': return 'ODFNEW'
        if modtype.lower()[0] == 'a': return 'AODFNEW'
    
    inmodel = raw_input("Please give model parameters: teff  logg  feh  vmicro _OR_ model_filename\n")
    inmodel = inmodel.split()

    if len(inmodel) == 1:
        mod_fname, = get_filename("Please give MOOG model file",'r',inmodel[0])
        if os.path.abspath(mod_fname) != os.path.abspath('./FINALMODEL'): os.system("cp "+mod_fname+" ./FINALMODEL")
    else:
        teff,logg,feh,vmicro =  np.array(inmodel[:4],dtype=float)
        modtype = raw_input("Please give model type: "+", ".join(['KUROLD', 'ODFNEW', 'AODFNEW', 'NOVER'])+"\n")
        modtype = convert_modtype(modtype)
        while True:
            check = create_kurucz_atmo_model(teff,logg,feh,vmicro,modtype,filename='FINALMODEL',verbose=True)
            if not check:
                inmodel = raw_input("Please retry model params: teff logg feh vmicro model_type:")
                try:
                    teff,logg,feh,vmicro =  np.array(inmodel[:4],dtype=float)
                    modtype = convert_modtype(inmodel[4])
                except: pass
            else: break
            
def convert_atlas12_for_moog (atlas_out_file,new_moog_file=None,clobber=False,include_depth_vt=False,logepsH=12,ions=[0,1],elements=[]):
    """
    This takes the model output from ATLAS12 and converts it to MOOG's KURUCZ format for stellar atmospheres
    
    ================    =============================================================
    Keyword             (type) Description
    ================    =============================================================
    atlas_out_file      (str) Gives the file name of the final model from ATLAS12
    new_moog_file       (str/None) If str then uses this to create the new model file 
                           which MOOG can read
    clobber             (bool) if True then it will overwrite existing output files
    include_depth_vt    (bool) if True then it will include Vturb for all the model
                           depths in a way MOOG can read
    logepsH             (float) Fixes the abundance of H used. Only relavent for He
    ions                (int array) This array gives the ions to be used when 
                           creating an output. e.g. [0,1,2] ===>  26.0, 26.1 and 26.2
    elements            (int array) If empty it will use elements which depart from
                         the input metallicity other wise adopts this list to include
    ================   =============================================================
    
    NOTE: When it creates the abundance table at the bottom of the MOOG file it only 
    uses those abundances which depart from solar and doesn't include H or He
        
    """

    #=========================================================================#
    # initial functions

    if new_moog_file is not None:
        if os.path.exists(new_moog_file) and not clobber: raise IOError("File already exists: '"+new_moog_file+"'")   

    batom = Batom()
    
    def np_vstack_append (arr,item):
        dtype = arr.dtype.name    
        if len(arr) == 0: arr = np.array([item],dtype=dtype)
        else: arr = np.vstack((arr,item))
        return arr

    #=========================================================================#
    # read in the atlas file
    fin = open(atlas_out_file)
    in_lines = fin.readlines()
    fin.close
     
    def read_abund_el (eline,header):
        # formatted read
        pz = int(eline[:5])
        el = eline[5:7]
        perc = 10**(float(eline[7:14]))
        del_sol = -1*float(eline[15:])
        
        if pz < 96:
            logeps = batom[pz][3]+del_sol
        else: logeps = del_sol
        # could also do this based on the total
        
        scaled_solar = 1
        if del_sol != header['feh']: scaled_solar = 0

        return [pz,el,perc,del_sol,logeps,scaled_solar]
      
    def read_header (sline):
        """
        Checks for these lines:
        
        TEFF   5000.  GRAVITY 2.20000 LTE 
        TITLE  [-4.3] VTURB=1.3  L/H=1.25 NOVER NEW ODF                                 
        OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 0 0
        CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00

        """
        if sline[0]=='TEFF': 
            header['teff'] = float(sline[1])
            header['grav'] = float(sline[3])
            header['type'] = sline[4]
            return True
        
        elif sline[0]=='TITLE':
            header['feh'] = float(sline[1].strip(']').replace('[',''))
            header['vt'] = float(sline[2].split('=')[1])
            header['l/h'] = sline[3].split('=')[1]
            header['type+']= ' '.join(sline[4:])
            return True
        
        elif sline[0] == 'OPACITY': 
            header['opacity'] = " ".join(sline)             
            return True
        
        elif sline[0] == 'CONVECTION':
            header['convect'] = ' '.join(sline)
            return True

        return False
          
    def read_abund_scale (sline):
        """
        For reading and containing lines:
        
        ABUNDANCE SCALE   1.00000 ABUNDANCE CHANGE 1 0.92140 2 0.07843
        ABUNDANCE CHANGE  3 -15.24  4 -14.94  5 -13.79  6  -6.52  7  -8.42  8  -5.91
        ABUNDANCE CHANGE  9 -11.78 10  -8.26 11 -10.01 12  -8.76 13  -9.87 14  -8.79
        .....
        ABUNDANCE CHANGE 93 -24.30 94 -24.30 95 -24.30 96 -24.30 97 -24.30 98 -24.30
        ABUNDANCE CHANGE 99 -24.30
        ABUNDANCE TABLE

        """
        
        
        if sline[0] != 'ABUNDANCE': return False
        if sline[1] == 'SCALE':
            # do something with these line?
            return True
        
        elif sline[1] == 'CHANGE':
            # do something with these lines?
            return True
        
        elif sline[1] == 'TABLE':
            to_read_table = True
            return to_read_table
        
        return False
            
    def read_abundance_table(line,Z,abunds,logepsH,header):
        if not reading_metals:
            sline = line.split()
            val_h = float(sline[1])
            val_he = float(sline[3])
            
            total = 10**(logepsH)/val_h
            logepsHe = np.log10(val_he*total)
            scaled_solar = 0
            if round(logepsHe,2) == batom[2][3]: scaled_solar=1

            abund_table[1] = [1,'H',val_h,0,logepsH,1]
            abund_table[2] = [2,'He',val_he,0,logepsHe,scaled_solar]

            abunds = np_vstack_append(abunds,[1,val_h,0,logepsH,1])
            abunds = np_vstack_append(abunds,[2,val_he,0,logepsHe,scaled_solar])
            return 3,abunds
        
        ds = 20
        el_split = [line[:ds],
                    line[ds:2*ds],
                    line[2*ds:3*ds],
                    line[3*ds:4*ds],
                    line[4*ds:]]
        
        N = len(el_split)
        for i in xrange(N): 
            if len(el_split[i]) == 0: continue
            out = read_abund_el(el_split[i],header)
            abund_table[Z+i] = out
            abunds = np_vstack_append(abunds,[out[0]]+out[2:])
        Z += N
        
        return Z, abunds
             
    def convert_abunds_for_moog(abunds,ions,elements):
        # use only the non scaled lines
        if len(elements) == 0:
            which = np.argwhere(abunds.T[-1]==1).T[0]
        else:
            which = xrange(len(abunds))
        ab_lines = []
        ab_line = ''

        el_i = 0

        
        for i in which:
            z = abunds[i][0]

            if len(elements) ==0: 
                if z in [1,2]: continue
            elif z not in elements: continue
            
            logeps = abunds[i][3]
            for ion_state in ions:
                z += ion_state/10.0
                
                val = "    "+" ".join([format(z,'>5.2f'),format(logeps,'>5.3f')])
                                
                if el_i%5 == 0 and el_i != 0:
                    ab_lines.append(ab_line)
                    ab_line = val
                else: ab_line += val
                el_i += 1
            
        if el_i%5 != 0: ab_lines.append(ab_line)
        return ab_lines, el_i
        
    header = {}
    abund_table = {}
    abunds = np.array([],dtype=float)
      
    layers_data = np.array([],dtype=float)
    
    reading_metals = False
    to_read_table = False
    i_layer = 0
    N_layers = -1
    Z = 1

    for line in in_lines:
        line = line.rstrip()
        sline = line.split()
        
        # skip blank lines
        if len(sline) == 0: continue
        
        # read first few lines into header
        elif read_header(sline): continue

        #if read_abund_scale(sline): continue
        # skip some lines here that are ABUNDANCE ????
        
        # The next lines should be the abunance table
        elif " ".join(sline) == 'ABUNDANCE TABLE': 
            to_read_table = True
            continue
        
        elif sline[0] == 'READ': 
            # this line separates the abundance table from the 
            # layers of the star
            # READ DECK6 72 RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB
            to_read_table = False
            N_layers = int(sline[2])
            continue
        
        if to_read_table: 
            Z, abunds = read_abundance_table(line,Z,abunds,logepsH,header)
            reading_metals = True # after the first line which is H and He
            continue
                
        elif i_layer < N_layers:
            # read N lines following:
            # READ DECK6 72 RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB
            i_layer += 1            
            layers_data = np_vstack_append(layers_data,np.array(sline,dtype=float))
            
        # should be two more lines I don't really care about:
        #   PRADK 8.3210E-01
        #   BEGIN                    ITERATION  17 COMPLETED
    ab_lines, Natoms = convert_abunds_for_moog(abunds,ions,elements)
    if new_moog_file is None:
        return abunds,abund_table,header,layers_data,
    #=========================================================================#
    # save out as KURUCZ format for MOOG
    lines = []
    lines.append('KURUCZ')
    steff = format(header['teff'],'<10.1f')
    sfeh = format(header['feh'],'<5.2f')
    hdr = [steff,
           format(header['grav'],'<5.2f'),
           sfeh,
           format(header['vt'],'<5.2f'),
           header['type'],
           header['type+']]
    
    lines.append("  ".join(hdr))
    lines.append(format(len(layers_data),'>15'))

    vt_lines = []
    for i in xrange(len(layers_data)):
        arr = layers_data[i]
        dat = [format(arr[0],'>10.5f'), # rhox
               format(arr[1],'>9.1f'), # T
               format(arr[2],'>9.3e'), # P_g
               format(arr[3],'>9.3e'), # N_e
               format(arr[4],'>9.3e')] # ABROSS =?= K_ross
        lines.append(" ".join(dat))
                
        vt_val = " "+format(arr[6],'>12.3e')
        if i == 0: 
            vt_line = vt_val 
            continue
        
        if i%6 == 0:
            vt_lines.append(vt_line)
            vt_line = vt_val 
        else: vt_line += vt_val 
      
    if i%6 != 0: vt_lines.append(vt_line)

    if include_depth_vt: lines += vt_lines
    else: lines.append(format(header['vt'],'>13.3f'))
    
    lines.append('NATOMS '+format(Natoms,'>8')+" "+format(sfeh,'>14'))
    if Natoms != 0: lines += ab_lines
    
    Nmols = 0
    lines.append('NMOL '+format(Nmols,'>10'))
    
    fout = open(new_moog_file,'w')
    fout.write("\n".join(lines)+"\n")
    fout.close()
    return abunds