import brian2 as b2
import numpy as np
from square import generate_stim_wave

class VirtualMEA:
    def __init__(self, left_neurons, right_neurons, left_spikes, right_spikes, network):
        self.left_neurons = left_neurons
        self.right_neurons = right_neurons
        self.left_spikes = left_spikes
        self.right_spikes = right_spikes
        self.network = network
        self.last_action = 0
        
    def read_neural_data_buffer(self, num_channels, buffer_size, client_socket=None, num_bytes_per_element=8):
        """Simulate reading from MEA by running the Brian2 simulation"""
        # Run simulation for a fixed duration
        self.network.run(20 * b2.ms)
        
        # Generate simulated voltage traces from spike data
        result = np.zeros((num_channels, buffer_size))
        
        # Convert spike data to continuous traces
        for i, neuron_idx in enumerate(range(len(self.left_neurons))):
            if neuron_idx in self.left_spikes.i:
                spike_times = self.left_spikes.t[self.left_spikes.i == neuron_idx]
                for t in spike_times:
                    # Create spike waveform
                    peak_idx = int((t/b2.ms) * buffer_size/20)
                    if 0 <= peak_idx < buffer_size:
                        result[i, peak_idx:min(peak_idx+5, buffer_size)] = 100  # Spike amplitude
        
        # Do the same for right neurons
        for i, neuron_idx in enumerate(range(len(self.right_neurons))):
            if neuron_idx in self.right_spikes.i:
                spike_times = self.right_spikes.t[self.right_spikes.i == neuron_idx]
                for t in spike_times:
                    peak_idx = int((t/b2.ms) * buffer_size/20)
                    if 0 <= peak_idx < buffer_size:
                        result[i + num_channels//2, peak_idx:min(peak_idx+5, buffer_size)] = 100
        
        return result
    
    def stimulate_neurons(self, pole_angle, pole_angular_velocity, reward, client_socket=None, duration=100):
        """Stimulate neurons based on pole angle and angular velocity"""
        # Generate stimulation pattern
        stim_wave = generate_stim_wave(pole_angle, pole_angular_velocity, duration)
        
        # Determine which group to stimulate based on angle
        if pole_angle < 0:  # Leaning left
            target_neurons = self.left_neurons
        else:  # Leaning right
            target_neurons = self.right_neurons
            
        # Convert stimulation wave to current
        stim_current = np.mean(stim_wave) * 1e-12 * b2.amp
        
        # Apply stimulation to target neurons
        target_neurons.I = stim_current
        
        # Record last action for future reference
        self.last_action = 0 if pole_angle < 0 else 1
        
    def extract_neuron_action(self, raw_neural_data, threshold=3):
        """Process neural data to determine action (0 or 1)"""
        # Count spikes in left and right regions
        left_count = len(self.left_spikes.i)
        right_count = len(self.right_spikes.i)
        
        # Reset spike monitors for next iteration
        self.left_spikes.count = []
        self.right_spikes.count = []
        
        # Determine action based on spike counts
        action = 0 if left_count > right_count else 1
        self.last_action = action
        return action