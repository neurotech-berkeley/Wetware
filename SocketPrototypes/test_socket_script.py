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
            
            return all_channel_data
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
