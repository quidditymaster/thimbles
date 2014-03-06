from widgets import *
import matplotlib.pyplot as plt
import threading
import time
try:
    import pygame.midi
    pygame.midi.init()
    from PySide.QtCore import *
    from PySide.QtGui import *
except:
    pass

class ButtonMidi(QObject):
    pressed = Signal()
    released = Signal()
    
    def __init__(self, cc_num):
        super(ButtonMidi, self).__init__()
        self.cc_num = cc_num
        self.press_state = False
    
    def _on_press(self):
        print "button pressed", self.cc_num
        #import pdb; pdb.set_trace()
        self.press_state = True
        self.pressed.emit()
    
    def _on_release(self):
        print "button released", self.cc_num
        self.press_state = False
        self.released.emit()

    def is_button(self):
        return True

class SlidingMidi(QObject):
    valueChanged = Signal(int)
    
    def __init__(self, cc_num, start_value=0):
        super(SlidingMidi, self).__init__()
        self.cc_num = cc_num
        self.value = start_value
    
    def is_button(self):
        return False
    
    def set_value(self, value):
        print "value changed", self.cc_num, value
        self.value = value
        self.valueChanged.emit(value)

class NanoKontrol2(threading.Thread):
    
    def __init__(self, polling_frequency=0.05):
        super(NanoKontrol2, self).__init__()
        self.polling_frequency = polling_frequency
        self._stop = threading.Event()
        self.init_controls()
        self.init_cc_dict()
        self.device_id = 0
        nanoKontrol_found = False
        for device_num in range(pygame.midi.get_count()):
            device_info = pygame.midi.get_device_info(device_num)
            if ("nanoKONTROL2" in device_info[1]) and (device_info[2] == 1):
                self.device_id = device_num
                nanoKontrol_found = True
                break
        if not nanoKontrol_found:
            raise Exception("nanoKontrol2 not found")
        
        self.nk_in = pygame.midi.Input(self.device_id)
    
    def init_controls(self):
        self.rewind = ButtonMidi(43)
        self.fast_forward = ButtonMidi(44)
        self.stop=ButtonMidi(42)
        self.play = ButtonMidi(41)
        self.record = ButtonMidi(45)
        
        self.track_back = ButtonMidi(58)
        self.track_advance = ButtonMidi(59)
        
        self.set_marker = ButtonMidi(60)
        self.marker_back = ButtonMidi(61)
        self.marker_advance = ButtonMidi(62)
        self.cycle = ButtonMidi(46)
        
        self.slider1 = SlidingMidi(0)
        self.slider2 = SlidingMidi(1)
        self.slider3 = SlidingMidi(2)
        self.slider4 = SlidingMidi(3)
        self.slider5 = SlidingMidi(4)
        self.slider6 = SlidingMidi(5)
        self.slider7 = SlidingMidi(6)
        self.slider8 = SlidingMidi(7)
        
        self.knob1 = SlidingMidi(16)
        self.knob2 = SlidingMidi(17)
        self.knob3 = SlidingMidi(18)
        self.knob4 = SlidingMidi(19)
        self.knob5 = SlidingMidi(20)
        self.knob6 = SlidingMidi(21)
        self.knob7 = SlidingMidi(22)
        self.knob8 = SlidingMidi(23)
        
        self.s1 = ButtonMidi(32)
        self.s2 = ButtonMidi(33)
        self.s3 = ButtonMidi(34)
        self.s4 = ButtonMidi(35)
        self.s5 = ButtonMidi(36)
        self.s6 = ButtonMidi(37)
        self.s7 = ButtonMidi(38)
        self.s8 = ButtonMidi(39)
        
        self.m1 = ButtonMidi(48)
        self.m2 = ButtonMidi(49)
        self.m3 = ButtonMidi(50)
        self.m4 = ButtonMidi(51)
        self.m5 = ButtonMidi(52)
        self.m6 = ButtonMidi(53)
        self.m7 = ButtonMidi(54)
        self.m8 = ButtonMidi(55)
        
        self.r1 = ButtonMidi(64)
        self.r2 = ButtonMidi(65)
        self.r3 = ButtonMidi(66)
        self.r4 = ButtonMidi(67)
        self.r5 = ButtonMidi(68)
        self.r6 = ButtonMidi(69)
        self.r7 = ButtonMidi(70)
        self.r8 = ButtonMidi(71)
        
    def init_cc_dict(self):
        self.unique_buttons = [self.rewind, self.fast_forward, self.stop, 
                               self.play, self.record, self.track_back, 
                               self.track_advance, self.set_marker, 
                               self.marker_back, self.marker_advance,
                               self.cycle]
        self.sliders = [self.slider1, self.slider2, self.slider3, self.slider4,
                        self.slider5, self.slider6, self.slider7, self.slider8]
        self.knobs = [self.knob1, self.knob2, self.knob3, self.knob4,
                      self.knob5, self.knob6, self.knob7, self.knob8]
        self.s_buttons = [self.s1, self.s2, self.s3, self.s4,
                          self.s5, self.s6, self.s7, self.s8]
        self.m_buttons = [self.m1, self.m2, self.m3, self.m4,
                          self.m5, self.m6, self.m7, self.m8]
        self.r_buttons = [self.r1, self.r2, self.r3, self.r4,
                          self.r5, self.r6, self.r7, self.r8]
        
        self.cc_dict = {}
        for clist in [self.unique_buttons, self.sliders, self.knobs, self.s_buttons, self.m_buttons, self.r_buttons]:
            for control in clist:
                self.cc_dict[control.cc_num] = control
    
    def run(self):
        while True:
            if self.stopped():
                print "calling quit"
                pygame.midi.quit()
                del self.nk_in
                return
            #poll the device to see if there input to process
            new_input = self.nk_in.poll()
            if new_input:
                #max device event buffer is 200 events
                events = self.nk_in.read(200)
                for event, event_time in events:
                    device_status, cc_num, value, junk = event
                    control = self.cc_dict.get(cc_num)
                    if control == None:
                        print "control number %d not known" % cc_num
                        continue
                    if control.is_button():
                        if value > 63:
                            control._on_press()
                        else:
                            control._on_release()
                    else:
                        control.set_value(value)
            time.sleep(self.polling_frequency)
    
    def slider_state_vec(self):
        return np.array([s.value for s in self.sliders])

    def knob_state_vec(self):
        return np.array([k.value for k in self.knobs])
    
    def stop_me(self):
        print "stop set"
        self._stop.set()
    
    def stopped(self):
        return self._stop.isSet()

class ProjectionDisplay(QWidget):
    
    def __init__(self, data, controller, parent=None):
        super(ProjectionDisplay, self).__init__(parent)
        #qvb = QVBoxLayout()
        #self.mpl_wid = MatplotlibWidget()
        #qvb.addWidget(self.mpl_wid)
        self.data = data
        self.controller = controller
        self.init_plots()

    def get_proj_vecs(self):
        slvec = self.controller.slider_state_vec()
        proj1 = slvec[:4]-63.0
        proj2 = slvec[4:]-63.0
        p1sum = np.sqrt(np.sum(proj1**2))
        if p1sum > 0:
            proj1 /= p1sum
        p2sum = np.sqrt(np.sum(proj2**2))
        if p2sum > 0:
            proj2 /= p2sum 
        return proj1, proj2
    
    def get_x_y(self):
        p1, p2 = self.get_proj_vecs()
        print p1, p2, "proj vecs"
        x = np.dot(self.data, p1)
        y = np.dot(self.data, p2)
        return x, y
    
    def init_plots(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        x, y = self.get_x_y()
        self.plot ,= self.ax.plot(x, y, marker="+", linestyle="none")
    
    def update_plots(self):
        print "update plots called"
        x, y = self.get_x_y()
        self.plot.set_data(x, y)
        self.fig.canvas.draw()
        #self.mpl_wid.draw()
    
    @Slot()
    def print_tada(self):
        print "tada"


if __name__ == "__main__":
    qap = QApplication([])
    
    nk2 = NanoKontrol2()
    nk2.start()
    
    data = np.loadtxt("for_midi_game.dat")
    data -= np.mean(data, axis=0)
    slvec = nk2.slider_state_vec()
    pwid = ProjectionDisplay(data, nk2)    
    
    #pwid.update_plots()
    def call_update():
        while True:
            pwid.update_plots()
            time.sleep(0.1)
    
    plot_thread = threading.Thread(target=call_update)
    plot_thread.start()

    #nk2.play.pressed.connect(pwid.update_plots)
    #nk2.stop.pressed.connect(nk2.stop_me)
    plt.show()
    
    qap.exec_()
    
