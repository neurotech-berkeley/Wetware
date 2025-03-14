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
        self.last_action = 0
        self.num_channels = 60  # Assuming total number of MEA channels is 60
        self.failure_count = 0 # Tracking number of failures

    def stimulate_neurons(self, pole_angle, pole_angular_velocity, reward, client_socket, duration=100):
        """
        Generate stimulation patterns based on pole angle, angular velocity, and reward/punishment.
        Stimulate left or right channels selectively.
        """

        # Determine which group to stimulate based on action
        if np.abs(pole_angle) < 0.262:
            # Positive reward: Generate predictable reinforcing stimulation pattern
            stim_wave = generate_stim_wave(pole_angle, pole_angular_velocity, duration)
            active_group = "reward"
        else:
            # Negative reward or punishment: Generate random noise as unpredictable feedback
            stim_wave = self.generate_random_noise(duration)
            active_group = "punishment"
            self.failure_count += 1

        # Send stimulation selectively to left or right group based on last action
        
        if self.last_action == 0:
            selected_channels = range(self.num_channels // 2)  # Left channels
        else:
            selected_channels = range(self.num_channels // 2, self.num_channels)  # Right channels
        # Send waveform only to selected channels
        self.send_wave_to_selected_neurons(stim_wave, selected_channels)


    def generate_random_noise(self, duration, sampling_rate=500, base_voltage_amp=150, failure_count = 1):
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
        t = np.linspace(0, duration / 1000.0, num_samples)
        
        # Increase voltage amplitude based on the number of failures (caps at 2x base amp)
        max_amp_multiplier = min(2, 1 + 0.1 * failure_count)  
        voltage_amp = base_voltage_amp * max_amp_multiplier
        
        # Introduce frequency randomness based on failure count
        base_freq = 5 
        max_freq = min(50, base_freq + 3 * failure_count) 

        # Generate a chaotic oscillation pattern
        random_freq_1 = np.random.uniform(base_freq, max_freq)
        random_freq_2 = np.random.uniform(base_freq * 2, max_freq * 2)

        # Generate sine wave components with phase shifts to make the pattern unpredictable
        noise_wave = (
            voltage_amp * np.sin(2 * np.pi * random_freq_1 * t + np.random.uniform(0, 2*np.pi)) +
            0.5 * voltage_amp * np.sin(2 * np.pi * random_freq_2 * t + np.random.uniform(0, 2*np.pi))
        )
        
        # Add a uniform noise component for extra disruption
        noise_wave += np.random.uniform(-voltage_amp, voltage_amp, num_samples)
        
        # # Generate random values between -voltage_amp and +voltage_amp
        # random_noise = np.random.uniform(-voltage_amp, voltage_amp, num_samples)
    
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
    
    def send_wave_to_selected_neurons(self, wave, selected_channels):
        """
        Send a stimulation waveform to selected neurons.

        Parameters:
        - wave: The waveform to be sent (numpy array).
        - selected_channels: List of channels to which the waveform should be sent.
        """
        # Ensure the waveform is valid
        if not isinstance(wave, np.ndarray):
            raise ValueError("Waveform must be a numpy array.")

        # Convert waveform to bytes
        wave_bytes = wave.tobytes()

        # Create a socket connection to the stimulation server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Connect to the stimulation server
            client_socket.connect((self.server_ip, self.stimulation_port))

            # Send the selected channel indices first
            channel_indices = np.array(selected_channels, dtype=np.int32).tobytes()
            client_socket.sendall(channel_indices)

            # Send the waveform data
            client_socket.sendall(wave_bytes)

        except Exception as e:
            print(f"Error sending stimulation data: {e}")
        finally:
            # Close the socket connection
            client_socket.close()

    def extract_neuron_action(self, raw_neural_data, threshold=3):
        """
        Process raw neural data to extract an action (0 or 1) for CartPole.
        Split channels into left and right groups.
        """

        median_abs_deviations, abs_activity = MADs(
            np.linspace(0, 1, raw_neural_data.shape[-1]), raw_neural_data
        )
        spike_count = count_spikes(abs_activity, median_abs_deviations)

        num_channels = len(spike_count)
        left_spike_count = np.sum(spike_count[:num_channels // 2])  # First half of channels
        right_spike_count = np.sum(spike_count[num_channels // 2:])  # Second half of channels

        action = 0 if left_spike_count > right_spike_count else 1  # 0 for left, 1 for right
        self.last_action = action
        
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

                    
                    
