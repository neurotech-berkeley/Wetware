import clr
import sys
from System import Object, Int32
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

channels_in_block_value = 0

def monitor_mea_data():
    # Create device list and initialize it for MEA devices
    device_list = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
    # Check if devices are found
    usb_entries = device_list.GetUsbListEntries()
    if len(usb_entries) <= 0:
        print("No devices found!")
        return
    print(f"Found {len(usb_entries)} devices")
    # Connect to the first available device
    dacq = CMeaUSBDeviceNet()
    # Set up event handlers
    dacq.ChannelDataEvent += lambda sender, channel_handle, num_frames: handle_data_event(sender, channel_handle, num_frames)
    dacq.ErrorEvent += lambda msg, action: print(f"Error: {msg}")
    # Connect to the device
    status = dacq.Connect(usb_entries[0])
    if status != 0:
        print(f"Connection failed: {status}")
        return
    try:
        # Configure data acquisition
        dacq.StopDacq(0)  # Stop any ongoing acquisition
        # Configure the device similar to the C# example
        scu = CSCUFunctionNet(dacq)
        scu.SetDacqLegacyMode(False)
        samplerate = 50000
        dacq.SetSamplerate(samplerate, 0, 0)
        dacq.SetDataMode(DataModeEnumNet.Signed_32bit, 0)
        # For MEA2100-Mini with one headstage
        dacq.SetNumberOfAnalogChannels(60, 0, 0, 8, 0)
        dacq.EnableDigitalIn(DigitalDatastreamEnableEnumNet.DigitalIn |
                            DigitalDatastreamEnableEnumNet.DigitalOut |
                            DigitalDatastreamEnableEnumNet.Hs1SidebandLow |
                            DigitalDatastreamEnableEnumNet.Hs1SidebandHigh, 0)
        dacq.EnableChecksum(True, 0)
        # Create boxed objects for out parameters
        analog_channels = Int32(0)
        digital_channels = Int32(0)
        checksum_channels = Int32(0)
        timestamp_channels = Int32(0)
        channels_in_block = Int32(0)
        # Call GetChannelLayout with boxed objects as out parameters
        (analog_channels, digital_channels, checksum_channels, timestamp_channels, channels_in_block) = dacq.GetChannelLayout(analog_channels, digital_channels, checksum_channels, timestamp_channels, channels_in_block, 0)
        # Extract values from boxed objects
        analog_channels_value = int(analog_channels)
        digital_channels_value = int(digital_channels)
        checksum_channels_value = int(checksum_channels)
        timestamp_channels_value = int(timestamp_channels)
        global channels_in_block_value
        channels_in_block_value = int(channels_in_block)
        print(f"Channel layout: {analog_channels_value} analog, {digital_channels_value} digital, {channels_in_block_value} in block")
        # Configure channel block
        queue_size = samplerate
        threshold = samplerate // 100
        dacq.ChannelBlock.SetSelectedChannels(channels_in_block_value // 2, queue_size, threshold,
                                             SampleSizeNet.SampleSize32Signed,
                                             SampleDstSizeNet.SampleDstSize32,
                                             channels_in_block_value)
        # dacq.SetNumberOfAnalogChannels(60, 0, 0, 8, 0)
        dacq.ChannelBlock.SetCommonThreshold(threshold)
        dacq.ChannelBlock.SetCheckChecksum(checksum_channels_value, timestamp_channels_value)
        # # Start data acquisition
        dacq.StartDacq()
        print("Recording started. Press Ctrl+C to stop...")
        # Keep the program running to receive data events
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping recording...")
    finally:
        # Stop data acquisition and disconnect
        dacq.StopDacq(0)
        dacq.Disconnect()
        print("Disconnected from device")

def handle_data_event(dacq, cb_handle, num_frames):
    """Handle incoming data from the MEA device"""
    try:
        # Number of channels is channelsInBlock / 2 for 32-bit data
        global channels_in_block_value

        num_channels = channels_in_block_value // 2
        # Read data from each channel
        for i in range(num_channels):
            # For ReadFramesI32, we need a boxed object for the out parameter
            frames_ret = Int32(0)
            channel_data = dacq.ChannelBlock.ReadFramesI32(i, 0, num_frames, frames_ret)
            frames_ret_value = int(frames_ret)
            # Process the data as needed
            if i < 10:  # Just print first 10 channels to avoid console spam
                print(f"Channel {i}: Received {frames_ret_value} samples")
                if len(channel_data) > 0:
                    # Print first few values
                    print(f"  First values: {list(channel_data[0])[:5]}")
    except Exception as e:
        print(f"Error processing data: {e}")
if __name__ == "__main__":
    import time
    monitor_mea_data()