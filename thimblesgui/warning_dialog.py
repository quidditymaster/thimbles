from thimblesgui import QtGui
from thimblesgui.feedback import request_feedback
import traceback

class WarningDialog(QtGui.QDialog):
    stack_trace = None
    
    def __init__(self, message, error=None, parent=None):
        super(WarningDialog, self).__init__(parent)
        layout = QtGui.QGridLayout()
        self.message_label = QtGui.QLabel(message)
        layout.addWidget(self.message_label, 0, 0, 1, 3)
        if error != None:
            self.stack_trace = traceback.extract_stack()
            #self.stack_trace.pop(-1)
            self.error_text_box = QtGui.QPlainTextEdit()
            self.error_text_box.setPlainText(str(error))
            self.error_text_box.setReadOnly(True)
            layout.addWidget(self.error_text_box, 1, 0, 1, 3)
            self.trace_text_box = QtGui.QPlainTextEdit()
            self.trace_text_box.setPlainText(str(self.stack_trace))
            self.trace_text_box.setReadOnly(True)
            layout.addWidget(self.trace_text_box, 2, 0, 1, 3)
        self.feedback_btn = QtGui.QPushButton("send error message")
        layout.addWidget(self.feedback_btn, 3, 1, 1, 1)
        self.feedback_btn.connect(self.on_feedback)
        self.ok_btn = QtGui.QPushButton("acknowledged")
        lay.addWidget(self.ok_btn, 3, 2, 1, 1)
        self.setLayout(lay)
        self.ok_btn.clicked.connect(self.on_ok)
        
    def on_feedback(self):
        request_feedback()
    
    def warn(self):
        self.exec_()
    
    def on_ok(self):
        self.accept()

if __name__ == "__main__":
    qap = QtGui.QApplication([])    
    try:
        i, j = 1
    except TypeError as e:
        wd = WarningDialog(message="testing", error=e)
