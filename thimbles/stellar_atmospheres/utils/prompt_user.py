# These prompt the user for input on the command line

import os
import numpy as np
from copy import deepcopy

# ########################################################################### #

__all__ = ["get_bounds","get_choice","yesno","get_filename"]

# ########################################################################### #

def get_bounds (prompt,lower_too=False,default=(0,1e20),display_help='No Help Available'):
    while True:
        inbounds = raw_input(prompt)
        if inbounds.lower() in ['help','h','he']:
            print "-"*60
            print display_help
            print "-"*60
            print " "
            continue
        elif inbounds == '': return default
        else:
            spl = inbounds.strip().split()
            try:
                if len(spl)==1 and lower_too: return (float(spl[0]),1e10)
                else: return (float(spl[0]),float(spl[1]))
                
            except: pass
        print "Invalid input. Type 'help' for more info."
 
def get_choice (choices,question="Please pick one :",default=0,prompt=None):
    if not isinstance(choices,(list,np.ndarray,tuple)):
        raise TypeError("choices must be a list or array of choices")
    
    str_choices = np.array([str(x) for x in choices])
    
    reserved = ('a','abort','help','?','h')
    for val in reserved:
        if val in str_choices:
            raise ValueError("Sorry, '"+val+"' can't be a choice because it's reserved in "+", ".join(reserved))    
    try:
        str_choices[default]
    except IndexError:
        raise ValueError("default must be the index of the default choice in choices")
      
    choices_string = ", ".join(str_choices[:-1])+" or "+str_choices[-1]
            
    if prompt is None:
        prompt =  "".join((question,"  ",choices_string,'\n'))

    while True:
        choice = raw_input(str(prompt))
        choice = choice.replace("'","")
        if not choice:
            return np.array([default])
        
        if choice in ('a','abort'):
            return np.where(str_choices=='a')[0]
        
        if choice in ('h','help','?'):
            print("This routine allows you to pick one of the following choices:")
            print("   "+choices_string+"\n")
            print("Other options:")
            print("-"*60)
            print("[a]bort - will break this loop")
            print("[h]elp  - will display this help screen")
            print("[Return]- is choice : "+str_choices[default])
            print("-"*60) 
            print("")
            continue       
        
        if choice in str_choices:
            return np.where(str_choices==choice)[0]
        
        print("Please enter one of these choices : "+choices_string)

def get_yes_no (choice):
    if choice is None:
        return None
    if isinstance(choice,bool):
        if choice: return 'yes'
        else: return 'no'
    choice = choice.lower()
    if choice in ('no','n'):
        return 'no'
    elif choice in ('yes','ye','y'):
        return 'yes'

def yesno (question="Please choose:",default_answer='n',prompt=None):
    """
    Prompt for a yes or no question and return True or False respectively    
    
    Parameters
    ----------
    question : string
        This becomes the prompt with ('yes','no') appended to the end
    default_answer : 'y','yes',True or 'n','no',False
        if 'True' or 'y' then enter returns True, conversely for False and 'n'
    prompt : string or None
        If given this supersedes the prompt built using question and uses
        this string directly in the raw_input call 
        
    Returns
    -------
    yesno : boolean
        Returns 'True' for a 'yes' to the prompt and 'False' for a 'no'
        
    Notes
    -----
    __1)__ If default answer is given then the ('yes','no') will become ('yes',['no']) 
        or (['yes'],'no') with the [] giving the default value when enter is hit
    
    Examples
    --------
    >>> if yesno("Does 3/4==9/12? ","y"): 
    >>>     print "Yes!"*10
    Does 3/4==9/12? (['yes'],'no')
    y
    
    Yes! Yes! Yes! Yes! Yes! Yes! Yes! Yes! Yes! Yes!
    
    
    """ 
    default_answer = get_yes_no(default_answer)
    if prompt is None: 
        if   default_answer == 'no': 
            prompt = question+"(yes,[no])\n"
        elif default_answer == 'yes': 
            prompt = question+"([yes],no)\n"
        else: 
            prompt = question+"(yes,no)\n"
            
    # get the user answer
    while True:
        choice = raw_input(str(prompt)).lower()
        if choice in ('n','no') or (choice == '' and default_answer=='no'): 
            return False
        elif choice in ('y','ye','yes') or (choice == '' and default_answer=='yes'):
            return True
        else: 
            print "Please answer 'yes','no','y', or 'n'"

def get_filename (prompt='ENTER FILENAME:', iotype='r', default=(None,), enter_multi = False, filename=None, find_filename=False):
    """
    Interactively get a filename(s)
    
    This uses raw_input to collect information about filenames and returns the desired output.
    There are options to extract lists of files and to check a specific file before prompting
    
    
    Parameters
    ----------
    prompt : string
        Gives the string to be given when prompting for a filename
    iotype : 'r' or 'w'
        When 'r' this will perform checks to see if the file exists and return 
        the default value if not
        When 'w' this will perform a check to see if the file exists and 
        prompt whether you want to overwrite
    default : object
        This object is returned if no proper (see note 1) filename is found
    enter_multi : boolean
        If 'True' then it will prompt for multiple files to be entered
    filename : string or None
        If not None then it provides a filename which will first be checked
        then if it doesn't exist (for 'r') or exists (for 'w') then it will
        prompt the user appropriately
    find_filename : boolean
        If 'True' then if the file is not appropriate (see note 1) then it
        will further prompt the user for an appropriate value. If 'False'
        then it will display an message and return the default
        
    Returns
    -------
    filenames : tuple 
        Returns all the appropriate filenames which were collected.

    
    Notes
    -----
    __1)__ Appropriate entry means that if iotype if 'r' then the file should
        exist if iotype if 'w' then it shouldn't or if prompts for overwriting the file
    __2)__ This creates lists of the files using set so you won't get repeats of files
    
    
    Examples
    --------
    >>> fname, = get_filename()
    >>>
    >>> fnames = get_filename(enter_multi=True)
    
    
    """  
    base_prompt = deepcopy(prompt) 
    check_filename = (filename != None)
    files = set([])
      
    i = 0
        
    # Loop     
    while True:
        if enter_multi: 
            prompt = "["+format(i,"2")+"] "+base_prompt
        
        # Get the filename
        if check_filename: 
            fname = str(filename)
        else:            
            fname = raw_input(str(prompt))
            
            # view options
            if fname == '\t': 
                print "PWD>> "+os.path.abspath('.')
                print "-"*60
                os.system('ls')
                print ""
                continue
            # ignore blank lines
            if len(fname.split())==0: 
                if enter_multi: print "To stop entering files enter '.'"
                continue
            
            # spaced objects
            if len(fname.split()) > 1:
                print "Please only enter one file name at a time"
                continue
            
            # abort entering files
            if   fname in ('a','abort'): return default
            elif fname in ('h','help','?'):
                print "This routine allows you to enter files and will check to make sure you gave an appropriate response\n"
                if enter_multi: print "Entering multiple files, to stop enter filename '.'"
                
                print "The current default output is : "+str(default)
                print ""
                print "-"*60
                print "options:"
                print "[a]bort - will break the file enter loop"
                print "[h]elp  - will display this help screen"
                print "tab     - will display 'ls' of the PWD"
                print ".       - when entering multiple files this acts as an EOF"
                print ""
                continue
                
            if enter_multi and fname == '.': 
                break
        
        
        good_file = True
        
        # check if file exists and trying to write over
        if os.path.isfile(fname) and iotype == 'w':
            if check_filename: print "File given '"+fname+"'"
            if yesno("WARNING: File exists, OK to overwrite?",'n'): good_file = True
            else: good_file = False
                    
        elif not os.path.isfile(fname) and iotype == 'r': # file does not exist
            if check_filename: print "File given '"+fname+"'"
            print "File does not exist "
            good_file = False

            
        # you can only check the first filename given
        check_filename=False    
        
        # check 
        if good_file:
            if enter_multi and find_filename and i == 0 : 
                print "[ 0] gave file "+fname
             
            files.add(fname)
            i += 1
            
        # else try again
        elif not good_file and find_filename: 
            continue
        
        if not enter_multi: break

    return tuple(files)
