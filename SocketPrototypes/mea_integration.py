import clr
import sys
import numpy as np
import time
from System import Object, Int32, Array, Double
from threading import Thread, Event

# Add the path to where the McsUsbNet.dll is located
sys.path.append('MEASocketConnection/bin/x64/Debug/')  # Adjust this path
clr.AddReference('McsUsbNet')

# Import required namespaces from the DLL
from Mcs.Usb import (
    CMcsUsbListNet,
    DeviceEnumNet,
    CMeaUSBDeviceNet,
    CMcsUsbDacqNet,
    DataModeEnumNet,
    DigitalDatastreamEnableEnumNet,
    SampleSizeNet,
    SampleDstSizeNet,
    CSCUFunctionNet,
    CMcsUsbFactoryNet,
    CMcsUsbBStimNet
)

class IntegratedMEAInterface:
    def __init__(self):
        """Initialize the integrated MEA interface for direct hardware communication."""
        self.dacq = None
        self.bstim = None
        self.device_connected = False
        self.channels_in_block = 0
        self.num_channels = 60
        self.buffer_size = 100
        self.samplerate = 50000
        self.data_buffer = np.zeros((self.num_channels, self.buffer_size))
        self.data_ready = Event()
        self.recording_thread = None
        self.stop_recording = Event()
        self.last_action = 0
        self.failure_count = 0

    def connect_to_device(self):
        """Connect to the MEA device and configure it."""
        # Create device list and initialize it for MEA devices
        device_list = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
        
        # Check if devices are found
        usb_entries = device_list.GetUsbListEntries()
        if len(usb_entries) <= 0:
            print("No MEA devices found!")
            return False
            
        print(f"Found {len(usb_entries)} MEA devices")
        
        # Connect to the first available device for data acquisition
        self.dacq = CMeaUSBDeviceNet()
        
        # Set up event handlers
        self.dacq.ChannelDataEvent += self.handle_data_event
        self.dacq.ErrorEvent += lambda msg, action: print(f"Error: {msg}")
        
        # Connect to the device
        status = self.dacq.Connect(usb_entries[0])
        if status != 0:
            print(f"Connection to MEA device failed: {status}")
            return False
            
        # Configure the device for data acquisition
        self.configure_data_acquisition()
        
        # Connect to the same device for stimulation
        factory = CMcsUsbFactoryNet()
        self.bstim = factory.CreateBStim(usb_entries[0])
        if self.bstim is None:
            print("Failed to create stimulation interface")
            self.dacq.Disconnect()
            return False
            
        # Configure stimulation parameters
        self.configure_stimulation()
        
        self.device_connected = True
        return True
        
    def configure_data_acquisition(self):
        """Configure the device for data acquisition."""
        self.dacq.StopDacq(0)  # Stop any ongoing acquisition
        
        # Configure the device
        scu = CSCUFunctionNet(self.dacq)
        scu.SetDacqLegacyMode(False)
        
        self.dacq.SetSamplerate(self.samplerate, 0, 0)
        self.dacq.SetDataMode(DataModeEnumNet.Signed_32bit, 0)
        
        # For MEA2100-Mini with one headstage
        self.dacq.SetNumberOfAnalogChannels(60, 0, 0, 8, 0)
        self.dacq.EnableDigitalIn(DigitalDatastreamEnableEnumNet.DigitalIn |
                                 DigitalDatastreamEnableEnumNet.DigitalOut |
                                 DigitalDatastreamEnableEnumNet.Hs1SidebandLow |
                                 DigitalDatastreamEnumNet.Hs1SidebandHigh, 0)
        self.dacq.EnableChecksum(True, 0)
        
        # Create boxed objects for out parameters
        analog_channels = Int32(0)
        digital_channels = Int32(0)
        checksum_channels = Int32(0)
        timestamp_channels = Int32(0)
        channels_in_block = Int32(0)
        
        # Call GetChannelLayout with boxed objects as out parameters
        (analog_channels, digital_channels, checksum_channels, timestamp_channels, channels_in_block) = \
            self.dacq.GetChannelLayout(analog_channels, digital_channels, checksum_channels, 
                                      timestamp_channels, channels_in_block, 0)
        
        # Extract values from boxed objects
        analog_channels_value = int(analog_channels)
        digital_channels_value = int(digital_channels)
        checksum_channels_value = int(checksum_channels)
        timestamp_channels_value = int(timestamp_channels)
        self.channels_in_block = int(channels_in_block)
        
        print(f"Channel layout: {analog_channels_value} analog, {digital_channels_value} digital, "
              f"{self.channels_in_block} in block")
        
        # Configure channel block
        queue_size = self.samplerate
        threshold = self.samplerate // 100
        
        self.dacq.ChannelBlock.SetSelectedChannels(
            self.channels_in_block // 2, 
            queue_size, 
            threshold,
            SampleSizeNet.SampleSize32Signed,
            SampleDstSizeNet.SampleDstSize32,
            self.channels_in_block
        )
        
        self.dacq.ChannelBlock.SetCommonThreshold(threshold)
        self.dacq.ChannelBlock.SetCheckChecksum(checksum_channels_value, timestamp_channels_value)
    
    def configure_stimulation(self):
        """Configure the device for stimulation."""
        if self.bstim is None:
            print("Stimulation interface not available")
            return False
            
        # Initialize the stimulator
        status = self.bstim.Initialize()
        if status != 0:
            print(f"Failed to initialize stimulator: {status}")
            return False
            
        # Set up basic stimulation parameters
        self.bstim.SetVoltageMode()  # Use voltage mode for stimulation
        
        print("Stimulation interface configured successfully")
        return True
    
    def start_recording(self):
        """Start recording data from the MEA."""
        if not self.device_connected:
            print("Device not connected. Cannot start recording.")
            return False
            
        # Start data acquisition
        self.dacq.StartDacq()
        print("Recording started")
        
        # Start a thread to continuously process data
        self.stop_recording.clear()
        self.recording_thread = Thread(target=self.recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        return True
    
    def stop_recording(self):
        """Stop recording data from the MEA."""
        if self.recording_thread and self.recording_thread.is_alive():
            self.stop_recording.set()
            self.recording_thread.join(timeout=2.0)
            
        if self.device_connected:
            self.dacq.StopDacq(0)
            print("Recording stopped")
    
    def disconnect(self):
        """Disconnect from the MEA device."""
        self.stop_recording()
        
        if self.bstim:
            self.bstim.Disconnect()
            
        if self.dacq:
            self.dacq.Disconnect()
            
        self.device_connected = False
        print("Disconnected from MEA device")
    
    def handle_data_event(self, dacq, cb_handle, num_frames):
        """Handle incoming data from the MEA device."""
        try:
            # Number of channels is channelsInBlock / 2 for 32-bit data
            num_channels = self.channels_in_block // 2
            
            # Read data from each channel and store in buffer
            for i in range(min(num_channels, self.num_channels)):
                # For ReadFramesI32, we need a boxed object for the out parameter
                frames_ret = Int32(0)
                channel_data = dacq.ChannelBlock.ReadFramesI32(i, 0, num_frames, frames_ret)
                frames_ret_value = int(frames_ret)
                
                # Store the data in our buffer (take the last buffer_size samples)
                if frames_ret_value > 0:
                    data_array = np.array(list(channel_data[0]))
                    if len(data_array) >= self.buffer_size:
                        self.data_buffer[i] = data_array[-self.buffer_size:]
                    else:
                        # Pad with zeros if not enough data
                        pad_size = self.buffer_size - len(data_array)
                        self.data_buffer[i] = np.pad(data_array, (0, pad_size), 'constant')
            
            # Signal that new data is available
            self.data_ready.set()
            
        except Exception as e:
            print(f"Error processing MEA data: {e}")
    
    def recording_loop(self):
        """Background thread to continuously process incoming data."""
        while not self.stop_recording.is_set():
            # Wait for new data with timeout
            if self.data_ready.wait(timeout=0.1):
                self.data_ready.clear()
                # Process data if needed
                # (This is handled by read_neural_data_buffer when called)
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
    
    def read_neural_data_buffer(self):
        """Read the current neural data buffer."""
        # Return a copy of the current buffer
        return self.data_buffer.copy()
    
    def send_stimulation(self, waveform, selected_channels):
        """Send a stimulation waveform to selected channels."""
        if not self.device_connected or self.bstim is None:
            print("Device not connected or stimulation not available")
            return False
            
        try:
            # Convert numpy array to .NET array
            waveform = np.clip(waveform, -500, 500)
            num_samples = len(waveform)
            stim_data = Array.CreateInstance(Double, num_samples)
            for i in range(num_samples):
                stim_data[i] = float(waveform[i])
            
            # Apply stimulation to selected channels
            for channel in selected_channels:
                # Set the channel for stimulation
                self.bstim.SetStimulationPattern(channel, stim_data)
                
            # Trigger the stimulation
            self.bstim.StartStimulation()
            
            return True
            
        except Exception as e:
            print(f"Error sending stimulation: {e}")
            return False
    
    def extract_neuron_action(self, raw_neural_data, threshold=3):
        """Process raw neural data to extract an action (0 or 1) for CartPole."""
        from spike import MADs, count_spikes
        import numpy as np
        
        # Create time array for the data
        time_array = np.linspace(0, 1, raw_neural_data.shape[1])
        
        # Calculate median absolute deviations and activity
        median_abs_deviations, abs_activity, _, _ = MADs(time_array, raw_neural_data)
        
        # Count spikes based on threshold
        spike_count = count_spikes(abs_activity, median_abs_deviations, threshold)
        
        num_channels = len(spike_count)
        left_spike_count = np.sum(spike_count[:num_channels // 2])  # First half of channels
        right_spike_count = np.sum(spike_count[num_channels // 2:])  # Second half of channels
        
        action = 0 if left_spike_count > right_spike_count else 1  # 0 for left, 1 for right
        self.last_action = action
        
        return action
    
    def stimulate_neurons(self, pole_angle, pole_angular_velocity, reward):
        """Generate and apply stimulation based on pole angle, angular velocity, and reward."""
        from square import generate_stim_wave
        
        # Determine which pattern to use based on pole angle
        if np.abs(pole_angle) < 0.262:  # ~15 degrees
            # Positive reward: Generate predictable reinforcing stimulation pattern
            stim_wave = generate_stim_wave(pole_angle, pole_angular_velocity, 100)
            active_group = "reward"
        else:
            # Negative reward or punishment: Generate random noise as unpredictable feedback
            stim_wave = self.generate_random_noise(100)
            active_group = "punishment"
            self.failure_count += 1
        
        # Select channels based on last action
        if self.last_action == 0:
            selected_channels = range(self.num_channels // 2)  # Left channels
        else:
            selected_channels = range(self.num_channels // 2, self.num_channels)  # Right channels
        
        # Send the stimulation to selected channels
        self.send_stimulation(stim_wave, selected_channels)
    
    def generate_random_noise(self, duration, sampling_rate=500, base_voltage_amp=150):
        """Generate a random noise waveform for punishment."""
        num_samples = int(sampling_rate * (duration / 1000.0))
        t = np.linspace(0, duration / 1000.0, num_samples)
        
        # Increase voltage amplitude based on the number of failures (caps at 2x base amp)
        max_amp_multiplier = min(2, 1 + 0.1 * self.failure_count)
        voltage_amp = base_voltage_amp * max_amp_multiplier
        
        # Introduce frequency randomness based on failure count
        base_freq = 5
        max_freq = min(50, base_freq + 3 * self.failure_count)
        
        # Generate a chaotic oscillation pattern
        random_freq_1 = np.random.uniform(base_freq, max_freq)
        random_freq_2 = np.random.uniform(base_freq * 2, max_freq * 2)
        
        # Generate sine wave components with phase shifts
        noise_wave = (
            voltage_amp * np.sin(2 * np.pi * random_freq_1 * t + np.random.uniform(0, 2*np.pi)) +
            0.5 * voltage_amp * np.sin(2 * np.pi * random_freq_2 * t + np.random.uniform(0, 2*np.pi))
        )
        
        # Add a uniform noise component for extra disruption
        noise_wave += np.random.uniform(-voltage_amp, voltage_amp, num_samples)
        
        return noise_wave
