import numpy as np
import socket
import struct
from square import generate_stim_wave
from spike import MADs, count_spikes


class MCS_Device_Interface:
    def __init__(self, server_ip="127.0.0.1", stimulation_port=9090):
        """
        Initialize the MCS Device Interface with server IP and stimulation port.
        """
        self.server_ip = server_ip
        self.stimulation_port = stimulation_port

    def stimulate_neurons(self, pole_angle, pole_angular_velocity, reward, client_socket, duration=100):
        """
        Generate stimulation patterns based on pole angle, angular velocity, and reward/punishment,
        and send them to neurons via the stimulation server.

        Parameters:
        - pole_angle: The angle of the pole in the CartPole environment.
        - pole_angular_velocity: The angular velocity of the pole.
        - reward: The reward signal from the CartPole environment.
                Positive rewards reinforce behavior; negative rewards (or lack of reward) punish it.
        - client_socket: The socket connection to the stimulation server.
        - duration: Duration of stimulation in milliseconds (default: 100 ms).
        """
        if np.abs(pole_angle) < 0.10:
            # Positive reward: Generate a predictable reinforcing stimulation pattern
            stim_wave = generate_stim_wave(pole_angle, pole_angular_velocity, duration)
        else:
            # Negative reward or punishment: Generate random noise as unpredictable feedback
            stim_wave = self.generate_random_noise(duration)

        # Send the generated waveform to neurons
        self.send_wave_to_neurons(stim_wave)

    def generate_random_noise(self, duration, sampling_rate=500, voltage_amp=150):
        """
        Generate a random noise waveform for punishment.

        Parameters:
        - duration: Duration of the noise in milliseconds.
        - sampling_rate: Sampling rate for waveform generation (default: 500 Hz).
        - voltage_amp: Amplitude of the noise signal (default: 150 μV).

        Returns:
        - A numpy array representing the random noise waveform.
        """
        num_samples = int(sampling_rate * (duration / 1000.0))
        
        # Generate random values between -voltage_amp and +voltage_amp
        random_noise = np.random.uniform(-voltage_amp, voltage_amp, num_samples)
        
        return random_noise
    
    def send_wave_to_neurons(self, wave):
        """
        Send a stimulation waveform to neurons via the stimulation server.

        Parameters:
        - wave: The waveform to be sent (numpy array).
        """
        # Convert waveform to bytes
        wave_bytes = wave.tobytes()

        # Create a socket connection to the stimulation server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.server_ip, self.stimulation_port))

        # Send waveform data
        client_socket.sendall(wave_bytes)

        # Close the socket after sending
        client_socket.close()

    def extract_neuron_action(self, raw_neural_data, threshold=3):
        """
        Process raw neural data to extract an action (0 or 1) for CartPole.

        Parameters:
        - raw_neural_data: Neural data buffer (2D numpy array).
        - threshold: Spike count threshold to determine action.

        Returns:
        - action: 0 (move left) or 1 (move right).
        """
        # Process neural data to calculate spike counts
        median_abs_deviations, abs_activity = MADs(
            np.linspace(0, 1, raw_neural_data.shape[-1]), raw_neural_data
        )
        
        spike_count = count_spikes(abs_activity, median_abs_deviations)

        # Determine action based on spike count threshold
        action = 1 if np.sum(spike_count) > threshold else 0

        return action

    def recv_all(self, socket, n):
        """
        Receive exactly n bytes from a socket.

        Parameters:
        - socket: The socket connection.
        - n: Number of bytes to receive.

        Returns:
        - data: Received data as bytes.
        """
        data = b""
        while len(data) < n:
            packet = socket.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def read_neural_data_buffer(self, num_channels, buffer_size, client_socket, num_bytes_per_element=8):
        """
        Read neural data buffer from the MEA system.

        Parameters:
        - num_channels: Number of channels in the MEA system.
        - buffer_size: Number of samples per channel.
        - client_socket: Socket connection to receive data.
        - num_bytes_per_element: Number of bytes per data element (default: 8 for float64).

        Returns:
        - result: Neural data as a 2D numpy array (num_channels x buffer_size).
                Each element is a voltage reading from a specific channel at a specific time.
                Shape: (num_channels, buffer_size)
                Data type: float64
                Units: microvolts (uV).
        """
        # Initialize an empty array to hold the neural data
        result = np.empty((num_channels, buffer_size))

        # Receive the exact number of bytes needed for all channels and samples
        total_bytes = num_channels * buffer_size * num_bytes_per_element
        data = self.recv_all(client_socket, total_bytes)

        if data is None:
            raise ValueError("Failed to receive neural data from the socket.")

        # Parse the received data into a 2D numpy array
        arr = []
        for i in range(num_channels):
            temp = []
            for j in range(buffer_size):
                start_idx = (i * buffer_size + j) * num_bytes_per_element
                end_idx = start_idx + num_bytes_per_element
                temp.append(struct.unpack('d', data[start_idx:end_idx])[0])
            arr.append(temp)

        # Convert to numpy array and reshape to (num_channels, buffer_size)
        result = np.array(arr)
        return result

                    
                    
