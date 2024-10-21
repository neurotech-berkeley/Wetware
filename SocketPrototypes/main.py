import socket
from utils import read_bytes, write_bytes

num_channels = 60
buffer_size = 100

# Define the servers' IP address and port
server_ip = "127.0.0.1"  # Localhost
recording_port = 8080  # C# recording server port
stimulation_port = 9090 # python stimulation server port

def send_command(command):
    # connect to the C# client
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, stimulation_port))
    
    # send command to the C# client
    client_socket.sendall(command.encode())
    
    # close connection
    client_socket.close()

def start_stimulation():
    send_command("start")

def stop_stimulation():
    send_command("stop")

def main():
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, recording_port))

    # Main loop/controller
    while True:
        curr_timestep_data = read_bytes(num_channels, buffer_size, client_socket)

        # TODO signal processing + extract action
        # TODO OpenAI Gym environment interaction
        # 
