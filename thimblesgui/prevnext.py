
from thimblesgui import QtCore, QtGui, Qt

class PrevNext(QtGui.QWidget):
    prev = QtCore.Signal()
    next = QtCore.Signal()
    
    def __init__(self, duration=1, with_play=False, parent=None):
        super(PrevNext, self).__init__(parent)
        layout = QtGui.QGridLayout()
        self.prev_btn = QtGui.QPushButton("prev")
        self.next_btn = QtGui.QPushButton("next")
        self.with_play = with_play
        if with_play:
            self.duration = int(duration*1000)
            self.duration_le = QtGui.QLineEdit("%5.3f" % duration)
            self.timer = QtCore.QTimer(self)
            self.timer.start(self.duration)
            self.play_toggle_btn = QtGui.QPushButton("Play/Pause")
            self.play_toggle_btn.setCheckable(True)
            self.play_toggle_btn.setChecked(True)
            #layout
            layout.addWidget(self.duration_le, 1, 0, 1, 2)
            layout.addWidget(self.play_toggle_btn, 2, 0, 1, 2)
            #connect
            self.timer.timeout.connect(self.on_timeout)
            self.duration_le.editingFinished.connect(self.set_duration)
            self.play_toggle_btn.clicked.connect(self.toggle_pause)
        
        #layout
        layout.addWidget(self.prev_btn, 0, 0, 1, 1)
        layout.addWidget(self.next_btn, 0, 1, 1, 1)
        self.setLayout(layout)
        
        #connect
        self.prev_btn.clicked.connect(self.on_prev_clicked)
        self.next_btn.clicked.connect(self.on_next_clicked)
    
    @property
    def paused(self):
        if not self.with_play:
            return True
        return self.play_toggle_btn.isChecked()
    
    def pause(self):
        if self.with_play:
            self.play_toggle_btn.setChecked(True)
    
    def play(self):
        self.play_toggle_btn.setChecked(False)
    
    def on_prev_clicked(self):
        self.pause()
        self.emit_prev()
    
    def on_next_clicked(self):
        self.pause()
        self.emit_next()
    
    def emit_next(self):
        self.next.emit()
    
    def emit_prev(self):
        self.prev.emit()
    
    def on_timeout(self):
        #print "timer went off"
        if not self.paused:
            if self.duration > 0:
                self.emit_next()
            elif self.duration < 0:
                self.emit_prev()
    
    def set_duration(self):
        try:
            duration_text = self.duration_le.text()
            new_duration = int(float(duration_text)*1000)
            if abs(new_duration) < 10:
                raise Exception("duration too small")
            new_duration_success = True
        except:
            print("could not recognize new duration reverting to old")
            new_duration_success = False
            self.duration_le.setText("%5.5f" % self.duration)
        if new_duration_success:
            self.duration = new_duration
            self.timer.setInterval(abs(new_duration))
    
    def toggle_pause(self):
        if self.paused:
            self.pause()
        else:
            self.play()

if __name__ == "__main__":
    qap = QtGui.QApplication([])
    pn = PrevNext()#with_play=True)
    pn.show()
    qap.exec_()
