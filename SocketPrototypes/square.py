import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

angle_threshold = 24
velocity_threshold = 2
angular_velocity_threshold = 2

def angle_to_wave(angle, duration, sampling_rate=500, voltage_amp = 150,duty_frac=0.1, intra_duration = 0.02):
    t = np.linspace(0, duration, sampling_rate) # 500Hz sampling rate
    # for now, angle (-12, 12) maps linearly to (4, 40)
    square_wave_freq = 1.5 * angle + 22 # needed to create 5Hz square wave

    out1 = voltage_amp/2 * (sp.signal.square(2 * np.pi * square_wave_freq * t, duty=duty_frac) + 1)
    out2 = -voltage_amp/2 * (sp.signal.square(2 * np.pi * square_wave_freq * t, duty=duty_frac) + 1)
    intra_sample = int(intra_duration * sampling_rate)
    shifted_out2 = np.append(np.zeros(intra_sample), out2)[:len(out2)]
    ret = shifted_out2 + out1
    plt.plot(shifted_out2 + out1)
    return ret

def ang_vec_to_wave(ang_vec, duration, sampling_rate=500, voltage_amp = 150,duty_frac=0.1, intra_duration = 0.02):
    t = np.linspace(0, duration, sampling_rate) # 500Hz sampling rate
    # for now, angle (-12, 12) maps linearly to (4, 40)
    square_wave_freq = 9 * ang_vec + 22 # needed to create 5Hz square wave

    out1 = voltage_amp/2 * (sp.signal.square(2 * np.pi * square_wave_freq * t, duty=duty_frac) + 1)
    out2 = -voltage_amp/2 * (sp.signal.square(2 * np.pi * square_wave_freq * t, duty=duty_frac) + 1)
    intra_sample = int(intra_duration * sampling_rate)
    shifted_out2 = np.append(np.zeros(intra_sample), out2)[:len(out2)]
    ret = shifted_out2 + out1
    plt.plot(shifted_out2 + out1)
    return ret