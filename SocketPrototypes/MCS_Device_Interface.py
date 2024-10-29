from square import angle_to_wave
from spike import MADs, count_spikes
import numpy as np
import struct

class MCS_Device_Interface:

    def __init__(self):
        # Initialize necessary parameters if needed
        pass

    def send_pole_angle(self, pole_angle, duration=100):
        """
        Send the pole angle data to neurons by converting it into a stimulation signal.
        """
        # Generate the waveform or spike signal based on the pole_angle
        wave = angle_to_wave(pole_angle, duration)
        
        self.send_wave_to_neurons(wave)

    def send_wave_to_neurons(self, wave):
        pass

    # TODO change linspace stop bound to correct buffer length
    def extract_neuron_action(self, raw_neural_data, threshold=3):
        """
        Receive spike data after stimulation and return action
        """
        # time, data = self.get_raw_neuron_data()
        median_abs_deviations, abs_activity = MADs(np.linspace(0, 1, raw_neural_data.shape[-1]), raw_neural_data)
        # TODO fix logic
        spike_count = count_spikes(abs_activity, median_abs_deviations)
        action = 1 if spike_count > threshold else 0

        return action

    # reading bytes from socket
    def recv_all(self, socket, n):
        data = b''
        while len(data) < n:
            packet = socket.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def read_neural_data_buffer(self, numChannels, bufferSize, client_socket, num_bytes_per_element=8):
        result = np.empty(numChannels * bufferSize)

        data = self.recv_all(client_socket, numChannels*bufferSize*num_bytes_per_element)

        arr = []
        # Receive data from the server
        for i in range(numChannels):
            temp = []
            for j in range(bufferSize):
                temp += [struct.unpack('d', data[(i*bufferSize*num_bytes_per_element)+j*num_bytes_per_element:(i*bufferSize*num_bytes_per_element)+(j+1)*num_bytes_per_element])[0]]
            arr += [temp]

        result = np.array(arr)
        
        return result.reshape((numChannels, bufferSize))