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
def test_single_channel(channel_number=0):
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
        print(f"Reading data from channel {channel_number}...")
        num_channels = 60
        buffer_size = 100
        num_bytes_per_element = 8  # 8 bytes for double
        total_bytes = num_channels * buffer_size * num_bytes_per_element
        data = recv_all(client_socket, total_bytes)
        if data:
            print(f"Received {len(data)} bytes of data")
            # Extract data for the specified channel
            channel_data = []
            for j in range(buffer_size):
                start_idx = (channel_number * buffer_size + j) * num_bytes_per_element
                end_idx = start_idx + num_bytes_per_element
                value = struct.unpack('d', data[start_idx:end_idx])[0]
                channel_data.append(value)
            # Print the first few values
            print(f"First 10 samples from channel {channel_number}:")
            for i, value in enumerate(channel_data[:10]):
                print(f"Sample {i}: {value}")
            # Plot the channel data
            plt.figure(figsize=(10, 6))
            plt.plot(channel_data)
            plt.title(f"Channel {channel_number} Data")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.savefig(f"channel_{channel_number}_data.png")
            plt.show()
            return channel_data
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
    # Change the channel number to test a specific channel (0-59)
    channel_to_test = 0
    test_single_channel(channel_to_test)