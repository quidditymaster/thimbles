#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE: For getting feedback from a user
AUTHOR: Dylan Gregersen
DATE: Fri Oct  3 15:45:21 2014
"""
# ########################################################################### #

# import modules 


import os 
import sys 
import re 
import time
from collections import OrderedDict
from PySide import QtCore,QtGui

# ########################################################################### #

# set the directory to put feedback files into. If an "" empty string then 
# no files will be saved
FEEDBACK_DIRECTORY = "" # os.path.join(os.path.dirname(__file__),"resources")

# set the default email list. If it's an empty list then no feedback is sent
FEEDBACK_EMAIL_LIST = [] #['quidditymaster@gmail.com']

# ########################################################################### #

class FeedbackForm (QtGui.QDialog):
    
    def __init__ (self, parent=None):
        QtGui.QDialog.__init__(parent)
        self.setWindowTitle('Thmimbles Feedback')
        self.initUI()
        self.feedback_text.setFocus()               
        self.is_accepted = False 
        self.setModal(True)
    
    def guess_username (self):
        """ guess who is submitting """
        # use whoami system command
        sysout = os.popen3("whoami")
        username = sysout[1].read().rstrip()

        # if the user has a u0123456 University of Utah Id then transform that to their name
        if re.search("u\d\d\d\d\d\d\d",username) is not None:
            grep_keyword = 'Name: '
    	    sysout = os.popen3("finger "+username+" | grep "+grep_keyword)
            finger_out = sysout[1].read().rstrip() #  stdin, stdout, stderr = sysout    
            i = finger_out.find(grep_keyword)
            if i >= 0:
                i += len(grep_keyword)
                username = finger_out[i:].strip().lower()
                
        return username
    
    def _init_feedback_type (self):        
        groupBox = QtGui.QGroupBox("Type of feedback")

        # types of feedback 
        self.feedback_types = {}        
        self.feedback_types['bug'] = QtGui.QRadioButton("&Bug")
        self.feedback_types['er'] = QtGui.QRadioButton("E&nhancement Request")

        # select one
        self.feedback_types['bug'].setChecked(True)

        # add feedback radio buttons
        hb = QtGui.QHBoxLayout()        
        for key in self.feedback_types:
            hb.addWidget(self.feedback_types[key])
        groupBox.setLayout(hb)
        return groupBox        

    def _init_feedback_text (self):
        groupBox = QtGui.QGroupBox("Please provide Feedback")
        vb = QtGui.QVBoxLayout()
        
        # Title for the feedback 
        label = QtGui.QLabel("Title ")
        self.feedback_title = QtGui.QLineEdit()
        hb = QtGui.QHBoxLayout()        
        hb.addWidget(label)
        hb.addWidget(self.feedback_title)      
        vb.addLayout(hb)
        
        # message from developer about how to use text box
        label = QtGui.QLabel(("If this is an error please describe what you "
        "were doing and how to recreate error (if possible). Also, if you can, copy and paste "
            "the Python traceback. Thanks!"))
        label.setWordWrap(True)        
        vb.addWidget(label)
        
        # main feedback text box                                            
        self.feedback_text = QtGui.QTextEdit() 
        vb.addWidget(self.feedback_text)
                            
        groupBox.setLayout(vb)
        return groupBox                

    def add_error_message (self,error_msg):
        if not len(error_msg):
            return 
        
        delim = "**** CONTEXT ****"
        msg = "\n\n{delim}\n\n{msg}\n\n{delim}".format(delim=delim,msg=error_msg)
        text = self.feedback_text.toPlainText()+msg
        self.feedback_text.setPlainText(text)
        cursor = self.feedback_text.textCursor()
        cursor.setPosition(cursor.Start)
        
    def _init_priority_box (self):
        # what priority to give this
        groupBox = QtGui.QGroupBox("Priority Level")
        lay = QtGui.QHBoxLayout()        
        label = QtGui.QLabel("0 (min) -- 10 (max)")
        lay.addWidget(label)    
        self.priority_spinbox = QtGui.QSpinBox()
        self.priority_spinbox.setRange(0, 10)
        self.priority_spinbox.setSingleStep(1)
        self.priority_spinbox.setValue(5)
        lay.addWidget(self.priority_spinbox)
        groupBox.setLayout(lay)
        return groupBox      
        
    def initUI (self):
        mainlayout = QtGui.QVBoxLayout()                      
        
        # UserName        
        label = QtGui.QLabel("Name ")
        self.username_text = QtGui.QLineEdit()
        self.username_text.setText(self.guess_username())       
        hb = QtGui.QHBoxLayout()        
        hb.addWidget(label)
        hb.addWidget(self.username_text)
        mainlayout.addLayout(hb)

        # Type of problem : bug, enhancement,        
        mainlayout.addWidget(self._init_feedback_type())    
                
        # Text input 
        mainlayout.addWidget(self._init_feedback_text())
                
        # Priority : 1-10
        mainlayout.addWidget(self._init_priority_box())
                         
        # ok,cancel
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
              
        mainlayout.addWidget(buttonBox)      
        self.setLayout(mainlayout)
    
    def accept (self,*args,**kwargs):    
        self.is_accepted = True
        QtGui.QDialog.accept(self,*args,**kwargs)
    
    def get_results (self):
        """ Converts all information into a dictionary """
        results = OrderedDict()
        if not self.is_accepted:
            return results
        results['name'] = self.username_text.text()
        results['timestamp'] = time.time()
        results['ctimestamp'] = time.ctime()        
        results['title'] = self.feedback_title.text()
        results['text'] = self.feedback_text.toPlainText()
        for key in self.feedback_types:
            if self.feedback_types[key].isChecked():
                results['type'] = key
        results['priority'] = self.priority_spinbox.value()
        return results 
        
    def save_to_file (self,dirpath,filename=None):
        """ Save message to file
        
        Parameters
        dirpath : string
            Directory path
        filename : None or string
            If None then uses self.get_filename()        
        
        """
        if not os.path.isdir(dirpath):
            raise IOError("no such directory '{}'".format(dirpath))
        if filename is None:
            filename = self.get_filename()
        fp = os.path.join(dirpath,filename)            
        msg = self.get_feedback_message()
        with open(fp,'w') as f:
            f.write(msg)
        
    def get_filename (self):
        """ return filename formatted with key information """
        results = self.get_results().copy()
        results['timestamp'] = int(results['timestamp'])
        return "{type}_{priority}_{timestamp}.txt".format(**results)
                
    def get_feedback_message (self):
        fmt = """
# ======================= #

Name : {name}
ctimestamp : {ctimestamp}
timestamp : {timestamp}
type : {type}
priority : {priority}
Title : {title}

# ======================= #


{text}
        """
        return fmt.format(**self.get_results())
        
    def email_feedback (self,email):
        """ Send the feedback to an email """
        results = self.get_results()
        subject = "[THIMBLES-FB] {type} {priority} -- {title} {ctimestamp}".format(**results)
        body = self.get_feedback_message()
        # TODO: send email
        cmd = 'echo "{}" | mail -s "{}" {}'.format(body,subject,email)
        os.system(cmd)

def request_feedback (error_msg="",error_title="",dirpath=None,emails=None, parent=None):
    """ Create and handle dialog asking user for feedback """
    if dirpath is None:
        dirpath = FEEDBACK_DIRECTORY
    if emails is None:
        emails = FEEDBACK_EMAIL_LIST    
    ff = FeedbackForm(parent=parent)    
    # show and activate the window
    ff.show()
    ff.raise_()
    ff.activateWindow()
    # add extra information
    ff.add_error_message(error_msg)
    ff.feedback_title.setText(error_title)
    # exec
    if ff.exec_():
        if len(dirpath):        
            ff.save_to_file(dirpath)
        print(ff.get_feedback_message())
        for e in emails:
            ff.email_feedback(e)            
    return ff 

pass
# ########################################################################### #
# Test application: 
# ########################################################################### #

if __name__ == '__main__':  
    try:
        app = QtGui.QApplication([])
    except RuntimeError:
        app = QtGui.QApplication.instance()        
    results = request_feedback("you hit an error!","Error Title")
    
