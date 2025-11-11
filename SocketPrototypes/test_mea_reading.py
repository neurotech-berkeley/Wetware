import clr
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
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
        queue_size = self.samplerate * 10
        threshold = self.samplerate // 50
        
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
        if not self.device_connected:
            print("Device not connected. Cannot start recording.")
            return False
            
        # Initialize data storage
        self.data_buffer = [[] for _ in range(self.num_channels)]
        self.timestamps = []
        self.start_time = time.time()
        
        # Start data acquisition
        self.dacq.StartDacq()
        print("Recording started")
        
        # Start a thread to continuously process data
        self.stop_recording_flag.clear()
        self.recording_thread = Thread(target=self.recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        return True
        
    def recording_loop(self):
        while not self.stop_recording_flag.is_set():
            current_time = time.time() - self.start_time
            
            # Check if recording duration has been reached
            if current_time >= self.recording_duration:
                print(f"Reached recording duration of {self.recording_duration} seconds")
                self.stop_recording_flag.set()
                break
                
            # Wait for new data with timeout
            if self.data_ready.wait(timeout=0.1):
                self.data_ready.clear()
                
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
    def handle_data_event(self, dacq, cb_handle, num_frames):
        try:
            # Record timestamp
            current_time = time.time() - self.start_time
            self.timestamps.append(current_time)
            
            # Number of channels is channelsInBlock / 2 for 32-bit data
            num_channels = self.channels_in_block // 2
            
            # Read data from each channel and store
            for i in range(min(num_channels, self.num_channels)):
                frames_ret = Int32(0)
                channel_data = dacq.ChannelBlock.ReadFramesI32(i, 0, num_frames, frames_ret)
                frames_ret_value = int(frames_ret)
                
                if frames_ret_value > 0:
                    # Convert to numpy array and store
                    data_array = np.array(list(channel_data[0]))
                    self.data_buffer[i].extend(data_array)
                    
            # Signal that new data is available
            self.data_ready.set()
            
        except Exception as e:
            print(f"Error processing MEA data: {e}")
            
    def stop_recording(self):
        if self.recording_thread and self.recording_thread.is_alive():
            self.stop_recording_flag.set()
            self.recording_thread.join(timeout=2.0)
            
        if self.device_connected:
            self.dacq.StopDacq(0)
            print("Recording stopped")
            
    def disconnect(self):
        self.stop_recording()
        if self.dacq:
            self.dacq.Disconnect()
        self.device_connected = False
        print("Disconnected from MEA device")
        
    def plot_and_save_data(self):
        print("Plotting and saving data...")
        
        # Convert data to numpy arrays
        data_arrays = []
        for channel_data in self.data_buffer:
            if channel_data:  # Check if there's data
                data_arrays.append(np.array(channel_data))
            else:
                data_arrays.append(np.array([]))
                
        # Create time array based on sample rate
        if self.timestamps:
            total_duration = self.timestamps[-1]
            
            # Plot a subset of channels (e.g., first 6)
            plt.figure(figsize=(15, 10))
            for i in range(min(6, self.num_channels)):
                if len(data_arrays[i]) > 0:
                    # Create time array for this channel's data
                    channel_time = np.linspace(0, total_duration, len(data_arrays[i]))
                    plt.subplot(6, 1, i+1)
                    plt.plot(channel_time, data_arrays[i])
                    plt.title(f"Channel {i+1}")
                    plt.ylabel("Voltage (μV)")
                    
            plt.xlabel("Time (s)")
            plt.tight_layout()
            plt.savefig("mea_recording_channels.png")
            plt.close()
            
            # Plot average activity across all channels
            plt.figure(figsize=(15, 5))
            valid_channels = [i for i, arr in enumerate(data_arrays) if len(arr) > 0]
            if valid_channels:
                # Resample all channels to the same length
                min_length = min(len(data_arrays[i]) for i in valid_channels)
                resampled_data = np.array([data_arrays[i][:min_length] for i in valid_channels])
                avg_activity = np.mean(np.abs(resampled_data), axis=0)
                
                # Create time array
                time_array = np.linspace(0, total_duration, len(avg_activity))
                
                plt.plot(time_array, avg_activity)
                plt.title("Average Activity Across All Channels")
                plt.xlabel("Time (s)")
                plt.ylabel("Average Absolute Voltage (μV)")
                plt.grid(True)
                plt.savefig("mea_recording_average.png")
                plt.close()
                
            print("Plots saved as 'mea_recording_channels.png' and 'mea_recording_average.png'")
        else:
            print("No data recorded to plot")

def main():
    print("Starting MEA data recording for 2 minutes...")
    
    # Initialize MEA data recorder
    mea_recorder = MEADataRecorder()
    
    try:
        # Connect to MEA device
        if not mea_recorder.connect_to_device():
            print("Failed to connect to MEA device. Exiting.")
            return
            
        # Start recording from MEA
        if not mea_recorder.start_recording():
            print("Failed to start recording. Exiting.")
            return
            
        print(f"Recording for {mea_recorder.recording_duration} seconds...")
        
        # Wait for recording to complete
        while mea_recorder.recording_thread and mea_recorder.recording_thread.is_alive():
            time.sleep(1.0)
            elapsed = time.time() - mea_recorder.start_time
            print(f"Recording in progress: {elapsed:.1f}/{mea_recorder.recording_duration} seconds", end="\r")
            
        print("\nRecording completed!")
        
        # Plot and save the recorded data
        mea_recorder.plot_and_save_data()
        
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    except Exception as e:
        print(f"Error during recording: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        mea_recorder.disconnect()
        print("Recording script completed.")

if __name__ == "__main__":
    main()