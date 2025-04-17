import socket
import struct
import numpy as np
import matplotlib.pyplot as plt

def recv_all(socket, n):
    """Receive exactly n bytes from a socket."""
    data = b""
    while len(data) < n:
        packet = socket.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def test_all_channels():
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_ip = "127.0.0.1"  # Localhost
    recording_port = 8080    # C# recording server port
    
    try:
        # Connect to the server
        print(f"Connecting to {server_ip}:{recording_port}...")
        client_socket.connect((server_ip, recording_port))
        print("Connected to recording server!")
        
        # Read data from the server
        print("Reading data from all 60 channels...")
        num_channels = 60
        buffer_size = 100
        num_bytes_per_element = 4  # 4 bytes for int32 (not 8 for double)
        
        total_bytes = num_channels * buffer_size * num_bytes_per_element
        data = recv_all(client_socket, total_bytes)
        
        if data:
            print(f"Received {len(data)} bytes of data")
            
            # Extract data for all channels
            all_channel_data = []
            for channel in range(num_channels):
                channel_data = []
                for j in range(buffer_size):
                    start_idx = (channel * buffer_size + j) * num_bytes_per_element
                    end_idx = start_idx + num_bytes_per_element
                    value = struct.unpack('i', data[start_idx:end_idx])[0]  # 'i' for int32
                    channel_data.append(value)
                all_channel_data.append(channel_data)
            
            # Print the first few values from a few channels
            for channel in range(0, num_channels, 10):  # Print every 10th channel
                print(f"\nChannel {channel} data (first 5 samples):")
                for i in range(5):
                    print(f"Sample {i}: {all_channel_data[channel][i]}")
            
            # Plot data from all channels
            plt.figure(figsize=(15, 10))
            for channel in range(num_channels):
                plt.subplot(8, 8, channel + 1)
                plt.plot(all_channel_data[channel])
                plt.title(f"Ch {channel}")
                plt.xticks([])  # Hide x-axis ticks for cleaner display
            
            plt.tight_layout()
            plt.savefig("all_channels_data.png")
            plt.show()
            
            # Define scaling factors based on MEA2100 documentation
            amplifier_gain = 5  # Check your specific hardware (2 or 5)
            signal_voltage_range = 70  # ±70 mV is typical for MEA2100 systems
            adc_max_count = 2**23  # 24-bit ADC (signed)

            # Function to convert raw values to millivolts
            def convert_to_mv(raw_value, channel_type):
                if channel_type == "headstage":
                    # For channels 0-7 (headstage channels with raw ADC values)
                    return (raw_value / adc_max_count) * signal_voltage_range * 1000 / amplifier_gain
                elif channel_type == "digital":
                    # For channel 16 (showing ramp pattern)
                    return raw_value * 0.01  # Adjust scaling factor as needed
                else:
                    # For other channels already in voltage units
                    return raw_value * 1000  # Convert from V to mV

            # Apply conversion to all channels
            all_channel_data_mv = []
            for channel in range(num_channels):
                if channel < 8:
                    # Headstage channels (0-7)
                    channel_type = "headstage"
                elif channel == 16:
                    # Digital channel with ramp
                    channel_type = "digital"
                else:
                    # Other channels
                    channel_type = "other"
                
                channel_data_mv = [convert_to_mv(value, channel_type) for value in all_channel_data[channel]]
                all_channel_data_mv.append(channel_data_mv)

            # Plot data with consistent mV units
            plt.figure(figsize=(15, 10))
            for channel in range(num_channels):
                plt.subplot(8, 8, channel + 1)
                plt.plot(all_channel_data_mv[channel])
                plt.title(f"Ch {channel}")
                plt.ylabel("mV")
                plt.xticks([])
                
            plt.tight_layout()
            plt.savefig("all_channels_millivolts.png")
            plt.show()
            
            # Find the channel with your waveform generator (let's say channel 36)
            waveform_channel = 36  # Change this to match your setup
            
            # Create a separate figure for the waveform channel
            plt.figure(figsize=(10, 6))
            plt.plot(all_channel_data[waveform_channel])
            plt.title(f"Channel {waveform_channel} - Expected 4 Hz Square Wave")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude (raw units)")
            plt.grid(True)
            plt.savefig(f"channel_{waveform_channel}_waveform.png")
            plt.show()
            
            # Plot the converted mV data for the waveform channel
            plt.figure(figsize=(10, 6))
            plt.plot(all_channel_data_mv[waveform_channel])
            plt.title(f"Channel {waveform_channel} - 4 Hz Square Wave (mV)")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude (mV)")
            plt.grid(True)
            plt.savefig(f"channel_{waveform_channel}_mV.png")
            plt.show()
            
            return all_channel_data_mv
        else:
            print("No data received")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None
        
    finally:
        # Close the socket
        client_socket.close()
        print("Connection closed")

if __name__ == "__main__":
    test_all_channels()
