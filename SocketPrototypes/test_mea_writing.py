import clr
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from threading import Thread, Event
from System import Object, Int32, Array, Double

# Add the path to where the McsUsbNet.dll is located
sys.path.append('MEASocketConnection/bin/x64/Debug/')
clr.AddReference('McsUsbNet')

from Mcs.Usb import (
    CMcsUsbListNet,
    DeviceEnumNet,
    CMeaUSBDeviceNet,
    CMcsUsbFactoryNet,
    CMcsUsbBStimNet
)

class MEAStimulator:
    def __init__(self):
        self.bstim = None
        self.device_connected = False
        self.num_channels = 60
        self.stim_duration = 120  # 2 minutes in seconds
        
    def connect_to_device(self):
        device_list = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
        usb_entries = device_list.GetUsbListEntries()
        
        if len(usb_entries) <= 0:
            print("No MEA devices found!")
            return False
            
        print(f"Found {len(usb_entries)} MEA devices")
        
        factory = CMcsUsbFactoryNet()
        self.bstim = factory.CreateBStim(usb_entries[0])
        
        if self.bstim is None:
            print("Failed to create stimulation interface")
            return False
            
        status = self.bstim.Initialize()
        if status != 0:
            print(f"Failed to initialize stimulator: {status}")
            return False
            
        self.bstim.SetVoltageMode()  # Use voltage mode for stimulation
        
        self.device_connected = True
        print("Successfully connected to MEA device for stimulation")
        return True
        
    def generate_biphasic_square_wave(self, duration_ms=100, amplitude_uv=150, frequency_hz=20, duty_cycle=0.1):
        """Generate a standard biphasic square wave"""
        sampling_rate = 500  # 500 Hz
        num_samples = int(sampling_rate * (duration_ms / 1000.0))
        t = np.linspace(0, duration_ms / 1000.0, num_samples)
        
        # Generate positive phase
        positive_phase = amplitude_uv * np.where(
            np.mod(t, 1.0/frequency_hz) < (duty_cycle/frequency_hz),
            1.0, 0.0
        )
        
        # Generate negative phase (delayed by 20ms)
        delay_samples = int(0.02 * sampling_rate)  # 20ms delay
        negative_phase = -amplitude_uv * np.where(
            np.mod(np.roll(t, -delay_samples), 1.0/frequency_hz) < (duty_cycle/frequency_hz),
            1.0, 0.0
        )
        
        # Combine phases
        biphasic_wave = positive_phase + negative_phase
        
        return biphasic_wave
        
    def send_stimulation(self, waveform, selected_channels):
        """Send a stimulation waveform to selected channels."""
        if not self.device_connected or self.bstim is None:
            print("Device not connected or stimulation not available")
            return False
            
        try:
            # Convert numpy array to .NET array
            num_samples = len(waveform)
            stim_data = Array.CreateInstance(Double, num_samples)
            for i in range(num_samples):
                stim_data[i] = float(waveform[i])
                
            # Apply stimulation to selected channels
            for channel in selected_channels:
                self.bstim.SetStimulationPattern(channel, stim_data)
                
            # Trigger the stimulation
            self.bstim.StartStimulation()
            return True
            
        except Exception as e:
            print(f"Error sending stimulation: {e}")
            return False
            
    def run_stimulation_sequence(self):
        """Run a 2-minute stimulation sequence with standard biphasic waveforms."""
        if not self.device_connected:
            print("Device not connected. Cannot stimulate.")
            return False
            
        print(f"Starting stimulation sequence for {self.stim_duration} seconds...")
        
        start_time = time.time()
        stim_count = 0
        
        # Create a figure to save stimulation patterns
        plt.figure(figsize=(10, 6))
        
        try:
            # Generate the standard biphasic waveform
            biphasic_wave = self.generate_biphasic_square_wave(
                duration_ms=100,
                amplitude_uv=150,  # 150 μV amplitude (similar to DishBrain)
                frequency_hz=20,   # 20 Hz frequency
                duty_cycle=0.1     # 10% duty cycle
            )
            
            # Plot the waveform
            plt.plot(biphasic_wave)
            plt.title("Standard Biphasic Square Wave")
            plt.xlabel("Samples")
            plt.ylabel("Voltage (μV)")
            plt.grid(True)
            plt.savefig("biphasic_waveform.png")
            
            # Stimulate in a pattern for 2 minutes
            while time.time() - start_time < self.stim_duration:
                stim_count += 1
                
                # Alternate between different channel groups
                if stim_count % 3 == 0:
                    # Stimulate left channels
                    selected_channels = range(self.num_channels // 2)
                    group_name = "left"
                elif stim_count % 3 == 1:
                    # Stimulate right channels
                    selected_channels = range(self.num_channels // 2, self.num_channels)
                    group_name = "right"
                else:
                    # Stimulate random channels
                    selected_channels = np.random.choice(range(self.num_channels), 
                                                        size=self.num_channels // 4, 
                                                        replace=False)
                    group_name = "random"
                
                # Send stimulation
                success = self.send_stimulation(biphasic_wave, selected_channels)
                
                if success:
                    elapsed = time.time() - start_time
                    print(f"Stimulation {stim_count}: group={group_name}, elapsed={elapsed:.1f}s")
                
                # Wait between stimulations
                time.sleep(1.0)
                
            print("Stimulation sequence completed")
            return True
            
        except KeyboardInterrupt:
            print("\nStimulation interrupted by user")
            return False
        except Exception as e:
            print(f"Error during stimulation: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def disconnect(self):
        """Disconnect from the MEA device."""
        if self.bstim:
            self.bstim.Disconnect()
        self.device_connected = False
        print("Disconnected from MEA device")

def main():
    print("Starting MEA stimulation with standard biphasic square wave...")
    
    # Initialize MEA stimulator
    mea_stimulator = MEAStimulator()
    
    try:
        # Connect to MEA device
        if not mea_stimulator.connect_to_device():
            print("Failed to connect to MEA device. Exiting.")
            return
            
        # Run stimulation sequence
        mea_stimulator.run_stimulation_sequence()
        
    except Exception as e:
        print(f"Error during stimulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        mea_stimulator.disconnect()
        print("Stimulation script completed.")

if __name__ == "__main__":
    main()
