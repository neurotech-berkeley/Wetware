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

def angle_velocity_to_wave(angle_velocity, duration, sampling_rate=500, voltage_amp = 150,duty_frac=0.1, intra_duration = 0.02):
    t = np.linspace(0, duration, sampling_rate) # 500Hz sampling rate
    # for now, angle (-12, 12) maps linearly to (4, 40)
    square_wave_freq = 1.5 * angle_velocity + 22 # needed to create 5Hz square wave

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

def generate_stim_wave(angle, angular_velocity, duration, sampling_rate=500, voltage_amp=150, duty_frac=0.1, intra_duration=0.02):
    """
    Generate a single biphasic square wave that encodes both angle and angular velocity.
    
    Parameters:
    -----------
    angle : float
        Pole angle in degrees (-30 to 30)
    angular_velocity : float
        Angular velocity of the pole
    duration : int
        Duration of stimulation in ms
    sampling_rate : int
        Sampling rate in Hz (default: 500)
    voltage_amp : float
        Stimulation amplitude in microvolts (default: 150)
    duty_frac : float
        Duty cycle of square wave (default: 0.1)
    intra_duration : float
        Delay between phases in seconds (default: 0.02)
        
    Returns:
    --------
    numpy.ndarray
        Combined stimulation waveform
    """
    t = np.linspace(0, duration, sampling_rate)
    
    norm_angle = np.clip(angle / 12, -1, 1)  
    norm_velocity = np.clip(angular_velocity / 2, -1, 1)  
    
    base_freq = 20  
    freq_mod = 10   
    amp_mod = 0.2 
    
    stim_freq = base_freq + freq_mod * norm_angle
    
    final_amp = voltage_amp * (1 + amp_mod * norm_velocity)
    
    out1 = final_amp/2 * (sp.signal.square(2 * np.pi * stim_freq * t, duty=duty_frac) + 1)
    out2 = -final_amp/2 * (sp.signal.square(2 * np.pi * stim_freq * t, duty=duty_frac) + 1)
    
    intra_sample = int(intra_duration * sampling_rate)
    shifted_out2 = np.append(np.zeros(intra_sample), out2)[:len(out2)]
    
    combined_wave = shifted_out2 + out1
    
    return combined_wave

def plot_stim_wave(angle, angular_velocity, duration=100):
    """Helper function to visualize the stimulation wave"""
    wave = generate_stim_wave(angle, angular_velocity, duration)
    plt.figure(figsize=(12, 4))
    plt.plot(wave)
    plt.title(f'Stimulation Wave (Angle: {angle:.1f}°, Angular Velocity: {angular_velocity:.1f} rad/s)')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (μV)')
    plt.grid(True)
    plt.show()