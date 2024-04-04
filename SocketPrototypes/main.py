import socket
from utils import read_bytes, write_bytes

num_channels = 60
buffer_size = 100

# Define the server's IP address and port
server_ip = "127.0.0.1"  # Localhost
server_port = 8080  # Example port number


def main():
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))

    # Main loop/controller
    while True:
        curr_timestep_data = read_bytes(num_channels, buffer_size, client_socket)

        # TODO signal processing + extract action
        # TODO OpenAI Gym environment interaction
        # 