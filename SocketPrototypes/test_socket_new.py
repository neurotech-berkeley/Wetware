import clr
import sys
import time

sys.path.append('MEASocketConnection/bin/x64/Debug/')  # Adjust this path
clr.AddReference('McsUsbNet')

from Mcs.Usb import CMcsUsbListNet, DeviceEnumNet, CMcsUsbDacqNet

def monitor_mea_data():
    # Create device list and initialize it
    deviceList = CMcsUsbListNet()
    deviceList.Initialize(DeviceEnumNet.MCS_MEA_DEVICE)
    
    # Check if devices are found
    if deviceList.Count <= 0:
        print("No devices found!")
        return
    
    # Connect to the first available device
    device = CMcsUsbDacqNet(deviceList.GetUsbListEntry(0))
    
    # Open the device
    device.Connect()
    
    try:
        # Configure data acquisition
        device.SetSamplerate(10000)  # 10 kHz sample rate
        
        # Enable channels you want to monitor
        for i in range(60):  # For a 60-electrode MEA
            device.EnableChannel(i, True)
        
        # Start data acquisition
        device.StartDacq()
        
        # Monitor data for a period of time
        start_time = time.time()
        duration = 10  # Monitor for 10 seconds
        
        while time.time() - start_time < duration:
            # Check if data is available
            if device.GetDataAvailable() > 0:
                # Read data from the device
                data = device.ReadData()
                
                # Process and display the data
                print(f"Read {len(data)} samples")
                
                # Example: Print first few values from first channel
                if len(data) > 0:
                    print(f"Channel 0 first 5 values: {data[0][:5]}")
            
            time.sleep(0.1)  # Small delay to prevent CPU overuse
        
        # Stop data acquisition
        device.StopDacq()
        
    finally:
        # Disconnect from the device
        device.Disconnect()

if __name__ == "__main__":
    monitor_mea_data()