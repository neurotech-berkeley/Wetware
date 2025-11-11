import clr
import sys
import time
from System import Int32

# Add the path to where the McsUsbNet.dll is located
dll_path = 'MEASocketConnection/bin/x64/Debug/'  # Adjust as needed
if dll_path not in sys.path:
    sys.path.append(dll_path)
try:
    clr.AddReference('McsUsbNet')
except Exception as e:
    print(f"Failed to load McsUsbNet DLL: {e}")
    sys.exit(1)

# Import required namespaces from the DLL
from Mcs.Usb import (
    CMcsUsbListNet,
    DeviceEnumNet,
    CMeaUSBDeviceNet,
    DataModeEnumNet,
    DigitalDatastreamEnableEnumNet,
    SampleSizeNet,
    SampleDstSizeNet,
    CSCUFunctionNet
)

channels_in_block_value = 0

def handle_data_event(sender, channel_handle, num_frames):
    """Handle incoming data from the MEA device"""
    global channels_in_block_value
    num_channels = channels_in_block_value // 2
    try:
        for i in range(num_channels):
            frames_ret = Int32(0)
            channel_data = sender.ChannelBlock.ReadFramesI32(i, 0, num_frames, frames_ret)
            frames_ret_value = int(frames_ret)
            if i < 10:
                print(f"Channel {i}: Received {frames_ret_value} samples")
                if len(channel_data) > 0:
                    # Print first few values
                    print(f"  First values: {list(channel_data[0])[:5]}")
    except Exception as e:
        print(f"Error processing data: {e}")

def monitor_mea_data():
    global channels_in_block_value

    # Create device list and initialize it for MEA devices
    device_list = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
    usb_entries = device_list.GetUsbListEntries()
    if len(usb_entries) <= 0:
        print("No devices found!")
        return

    print(f"Found {len(usb_entries)} devices")
    dacq = CMeaUSBDeviceNet()

    # Attach event handlers
    dacq.ChannelDataEvent += handle_data_event
    dacq.ErrorEvent += lambda msg, action: print(f"Error: {msg}")

    status = dacq.Connect(usb_entries[0])
    if status != 0:
        print(f"Connection failed: {status}")
        return

    try:
        dacq.StopDacq(0)
        scu = CSCUFunctionNet(dacq)
        scu.SetDacqLegacyMode(False)
        samplerate = 50000
        dacq.SetSamplerate(samplerate, 0, 0)
        dacq.SetDataMode(DataModeEnumNet.Signed_32bit, 0)
        dacq.SetNumberOfAnalogChannels(60, 0, 0, 8, 0)
        dacq.EnableDigitalIn(
            DigitalDatastreamEnableEnumNet.DigitalIn |
            DigitalDatastreamEnableEnumNet.DigitalOut |
            DigitalDatastreamEnableEnumNet.Hs1SidebandLow |
            DigitalDatastreamEnableEnumNet.Hs1SidebandHigh, 0)
        dacq.EnableChecksum(True, 0)

        # Prepare boxed Int32s for out parameters
        analog_channels = Int32(0)
        digital_channels = Int32(0)
        checksum_channels = Int32(0)
        timestamp_channels = Int32(0)
        channels_in_block = Int32(0)

        (analog_channels, digital_channels, checksum_channels, timestamp_channels, channels_in_block) = dacq.GetChannelLayout(
            analog_channels, digital_channels, checksum_channels, timestamp_channels, channels_in_block, 0
        )

        analog_channels_value = int(analog_channels)
        digital_channels_value = int(digital_channels)
        checksum_channels_value = int(checksum_channels)
        timestamp_channels_value = int(timestamp_channels)
        channels_in_block_value = int(channels_in_block)

        print(f"Channel layout: {analog_channels_value} analog, {digital_channels_value} digital, {channels_in_block_value} in block")

        queue_size = samplerate
        threshold = samplerate // 100
        dacq.ChannelBlock.SetSelectedChannels(
            channels_in_block_value // 2, queue_size, threshold,
            SampleSizeNet.SampleSize32Signed,
            SampleDstSizeNet.SampleDstSize32,
            channels_in_block_value
        )
        dacq.ChannelBlock.SetCommonThreshold(threshold)
        dacq.ChannelBlock.SetCheckChecksum(checksum_channels_value, timestamp_channels_value)

        dacq.StartDacq()
        print("Recording started. Press Ctrl+C to stop...")

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping recording...")
    finally:
        dacq.StopDacq(0)
        dacq.Disconnect()
        print("Disconnected from device")

if __name__ == "__main__":
    monitor_mea_data()
