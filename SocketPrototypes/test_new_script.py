import clr
import sys
import numpy as np
import time
import gym
from threading import Thread, Event
from System import Object, Int32, Array, Double

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
    CSCUFunctionNet
)

class MEADataReader:
    def __init__(self):
        """Initialize the MEA data reader for direct hardware communication."""
        self.dacq = None
        self.device_connected = False
        self.channels_in_block = 0
        self.num_channels = 60
        self.buffer_size = 100
        self.samplerate = 50000
        self.data_buffer = np.zeros((self.num_channels, self.buffer_size))
        self.data_ready = Event()
        self.recording_thread = None
        self.stop_recording_flag = Event()

    def connect_to_device(self):
        """Connect to the MEA device and configure it for data acquisition."""
        # Create device list and initialize it for MEA devices
        device_list = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
        
        # Check if devices are found
        usb_entries = device_list.GetUsbListEntries()
        if len(usb_entries) <= 0:
            print("No MEA devices found!")
            return False
        
        print("Number of MEA devices found:", len(usb_entries))
        
        # Connect to the first available device for data acquisition
        self.dacq = CMeaUSBDeviceNet()
        
        # Set up event handlers
        self.dacq.ChannelDataEvent += self.handle_data_event
        self.dacq.ErrorEvent += lambda msg, action: print("Error:", msg)
        
        # Connect to the device
        status = self.dacq.Connect(usb_entries[0])
        if status != 0:
            print(f"Connection to MEA device failed: {status}")
            return False
            
        # Configure the device for data acquisition
        self.configure_data_acquisition()
        
        self.device_connected = True
        print("Successfully connected to MEA device")
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
                                 DigitalDatastreamEnableEnumNet.Hs1SidebandHigh, 0)
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
    
    def start_recording(self):
        """Start recording data from the MEA."""
        if not self.device_connected:
            print("Device not connected. Cannot start recording.")
            return False
            
        # Start data acquisition
        self.dacq.StartDacq()
        print("Recording started")
        
        # Start a thread to continuously process data
        self.stop_recording_flag.clear()
        self.recording_thread = Thread(target=self.recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        return True
    
    def stop_recording(self):
        """Stop recording data from the MEA."""
        if self.recording_thread and self.recording_thread.is_alive():
            self.stop_recording_flag.set()
            self.recording_thread.join(timeout=2.0)
            
        if self.device_connected:
            self.dacq.StopDacq(0)
            print("Recording stopped")
    
    def disconnect(self):
        """Disconnect from the MEA device."""
        self.stop_recording()
        
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
        while not self.stop_recording_flag.is_set():
            # Wait for new data with timeout
            if self.data_ready.wait(timeout=0.1):
                self.data_ready.clear()
                # Process data if needed
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
    
    def read_neural_data_buffer(self):
        """Read the current neural data buffer."""
        # Return a copy of the current buffer
        return self.data_buffer.copy()
    
    def extract_action_from_constant_voltage(self):
        """
        Extract an action from constant voltage data.
        Since we're supplying constant voltage without real neurons,
        we'll use a simple threshold-based approach on the raw voltage values.
        """
        raw_data = self.read_neural_data_buffer()
        
        # Simple approach: compare average voltage in left vs right channels
        # Assuming first half of channels are "left" and second half are "right"
        left_channels = raw_data[:self.num_channels//2]
        right_channels = raw_data[self.num_channels//2:]
        
        left_avg = np.mean(np.abs(left_channels))
        right_avg = np.mean(np.abs(right_channels))
        
        # 0 for left, 1 for right
        action = 0 if left_avg > right_avg else 1
        
        return action

class CartPoleSimulation:
    def __init__(self, mea_reader):
        """Initialize the CartPole simulation with the MEA data reader."""
        self.env = gym.make('CartPole-v1', render_mode='human')
        self.mea_reader = mea_reader
        self.total_reward = 0
        
    def run_episode(self, max_steps=500):
        """Run a single episode of the CartPole simulation."""
        observation, _ = self.env.reset()
        self.total_reward = 0
        
        for step in range(max_steps):
            # Render the environment
            self.env.render()
            
            # Get action from MEA data
            action = self.mea_reader.extract_action_from_constant_voltage()
            
            # Take a step in the environment
            observation, reward, terminated, truncated, _ = self.env.step(action)
            
            # Update total reward
            self.total_reward += reward
            
            # Print state information
            pole_angle = observation[2]
            pole_angular_velocity = observation[3]
            print(f"Step {step}: Action={action}, Pole Angle={pole_angle:.4f}, "
                  f"Angular Velocity={pole_angular_velocity:.4f}, Reward={reward}")
            
            # Check if episode is done
            if terminated or truncated:
                print(f"Episode finished after {step+1} steps with total reward: {self.total_reward}")
                break
                
        return self.total_reward
        
    def close(self):
        """Close the environment."""
        self.env.close()

def main():
    """Main function to run the test script."""
    print("Starting MEA to CartPole test script...")
    
    # Initialize MEA data reader
    mea_reader = MEADataReader()
    
    try:
        # Connect to MEA device
        if not mea_reader.connect_to_device():
            print("Failed to connect to MEA device. Exiting.")
            return
        
        # Start recording from MEA
        if not mea_reader.start_recording():
            print("Failed to start recording. Exiting.")
            return
        
        # Wait a bit to ensure we have data in the buffer
        print("Waiting for initial data collection...")
        time.sleep(2.0)
        
        # Initialize CartPole simulation
        cart_pole = CartPoleSimulation(mea_reader)
        
        # Run a single episode
        print("Starting CartPole episode...")
        reward = cart_pole.run_episode()
        
        print(f"Test completed with total reward: {reward}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            cart_pole.close()
        except:
            pass
        
        mea_reader.disconnect()
        print("Test script completed.")

if __name__ == "__main__":
    main()
