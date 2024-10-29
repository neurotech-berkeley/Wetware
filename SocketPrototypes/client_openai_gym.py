import socket
from OpenAIGymAPI import OpenAIGymAPI
from MCS_Device_Interface import MCS_Device_Interface
# from utils import read_bytes, write_bytes

num_channels = 60
buffer_size = 100

# Define the servers' IP address and port
server_ip = "127.0.0.1"  # Localhost
recording_port = 8080  # C# recording server port
stimulation_port = 9090 # python stimulation server port

def send_command(stimulation_pattern):
    # connect to the C# client
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, stimulation_port))
    
    # send command to the C# client
    client_socket.sendall(stimulation_pattern.tobytes())

def main():
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, recording_port))

    # Main loop/controller
    mcs_device_interface = MCS_Device_Interface()
    openai_gym_api = OpenAIGymAPI(mcs_device_interface, num_channels, buffer_size)
    
    while True:
        # receiving neural data, extracting features, step in OpenAI gym
        pole_angle, pole_angular_velocity, reward, terminated = openai_gym_api.run_single_frame(client_socket)
        
        # generating stimulation pattern, stimulating neurons
        mcs_device_interface.stimulate_neurons(pole_angle, pole_angular_velocity, client_socket)
    
    # close connection
    client_socket.close()

main()