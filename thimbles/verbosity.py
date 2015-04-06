

# ########################################################################### #


from datetime import datetime
import os

from thimbles.options import Option, opts

class ThimblesLogger(object): 
    
    def __init__ (self):
        """controls printing messages to a log file and the screen
        """
        self._log_created = False
        self.file = None
        Option(name="logger", option_style="parent_dict")
        self._log_file = Option(name="log_file", parent="logger", help_="file to write logging values to", default=None)
        self._log_level = Option(name="log_level", parent="logger", help_="threshold for writing to log", default=10)
        self._print_level = Option(name="print_level", parent="logger", help_="threshold for printing", default=5)
        self._file_mode = Option(name="file_mode", parent="logger", help_="threshold for printing", default="w")
    
    @property
    def log_file(self):
        return self._log_file.value
    
    @property
    def log_level(self):
        return self._log_level.value
    
    @property
    def print_level(self):
        return self._print_level.value
    
    @property
    def file_mode(self):
        return self._file_mode.value
    
    def create_log(self):
        mode = self.file_mode
        if self.log_file is None:
            return
        try:
            if isinstance(self.log_file, str):
                self.file = open(self.log_file, mode)
                self._log_created = True
            else:
                raise TypeError("log_file should be a string or None")
            self.file.write("new log session {}\n".format(datetime.today().isoformat()))
        except Exception as e:
            print("attempting to open log file resulted in error {e} \n no log created".format(e))
            self.file = None
    
    def __call__ (self, msg , level=10):
        if not self._log_created:
            self.create_log() 
        if level <= self.print_level:
            print(msg) 
        if level <= self.log_level:
            if not self.file is None:
                self.log_file.write(msg + "\n")

logger = ThimblesLogger()