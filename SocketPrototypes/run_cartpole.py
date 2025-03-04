import socket
import numpy as np
from OpenAIGymAPI import OpenAIGymAPI
from MCS_Device_Interface import MCS_Device_Interface

# Define parameters
num_channels = 60
buffer_size = 100
server_ip = "127.0.0.1"  # Localhost
recording_port = 8080  # C# recording server port
stimulation_port = 9090  # Python stimulation server port
episodes = 100

def run_episodes():
    # Create a TCP/IP socket for communication with the MEA system
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, recording_port))

    # Initialize interfaces for MEA and CartPole
    mcs_device_interface = MCS_Device_Interface()
    openai_gym_api = OpenAIGymAPI(mcs_device_interface, num_channels, buffer_size)

    # Loop over episodes
    for episode in range(episodes):
        print(f"Starting Episode {episode + 1}/{episodes}")
        state, _ = openai_gym_api.env.reset()  # Reset CartPole environment for a new episode
        total_reward = 0
        done = False

        while not done:
            # Step 1: Run one frame of CartPole and get state variables and reward
            pole_angle, pole_angular_velocity, reward, terminated = openai_gym_api.run_single_frame(client_socket)

            # Step 2: Use reward/punishment to stimulate neurons accordingly
            mcs_device_interface.stimulate_neurons(pole_angle, pole_angular_velocity, reward, client_socket)

            # Step 3: Check termination condition
            done = terminated

        print(f"Episode {episode + 1} completed with total reward: {total_reward}")

    # Close connection after all episodes are completed
    client_socket.close()

if __name__ == "__main__":
    run_episodes()