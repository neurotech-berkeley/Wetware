# brian2_mea_simulator.py
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
from square import generate_stim_wave

class Brian2MEASimulator:
    """
    A class that simulates an MEA system using Brian2 spiking neural networks.
    Replaces the physical MEA interface with an in silico simulation.
    """
    
    def __init__(self, num_channels=60, buffer_size=100):
        """
        Initialize the Brian2 MEA simulator.
        
        Parameters:
        -----------
        num_channels : int
            Number of simulated electrode channels
        buffer_size : int
            Size of the data buffer for each channel
        """
        # Basic parameters
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        self.data_buffer = np.zeros((self.num_channels, self.buffer_size))
        self.device_connected = False
        self.last_action = 0
        self.failure_count = 0
        
        # Brian2 specific parameters
        defaultclock.dt = 0.1*ms  # Simulation time step
        self.network = None
        self.neurons = None
        self.synapses = None
        self.spike_monitor = None
        self.rate_monitor = None
        self.input_groups = []
        
        # Set up the network
        self.setup_network()
        self.device_connected = True
    
    def setup_network(self):
        """Set up the Brian2 spiking neural network."""
        # start_scope()
        
        # Define neuron model - Izhikevich model for biological realism
        neuron_eqs = '''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I)/ms : 1
        du/dt = (a*(b*v - u))/ms : 1
        I : 1
        a : 1
        b : 1
        c : 1
        d : 1
        '''
        
        # Create neuron groups - divide into left and right hemispheres
        N_per_side = self.num_channels // 2
        
        # Create the network with two groups (left and right)
        self.neurons = NeuronGroup(self.num_channels, neuron_eqs, 
                                  threshold='v>=30', reset='v=c; u+=d',
                                  method='euler')
        
        # Set parameters for regular spiking behavior
        self.neurons.a = 0.02
        self.neurons.b = 0.2
        self.neurons.c = -65
        self.neurons.d = 8
        
        # Randomize initial values
        self.neurons.v = -65
        self.neurons.u = self.neurons.b * self.neurons.v
        
        # Create excitatory connections within each hemisphere
        # Left hemisphere internal connections
        S_left = Synapses(self.neurons[:N_per_side], self.neurons[:N_per_side], 
                         'w : 1', on_pre='v_post += w')
        S_left.connect(p=0.1)  # 10% connectivity
        S_left.w = '0.5*rand()'
        
        # Right hemisphere internal connections
        S_right = Synapses(self.neurons[N_per_side:], self.neurons[N_per_side:], 
                          'w : 1', on_pre='v_post += w')
        S_right.connect(p=0.1)  # 10% connectivity
        S_right.w = '0.5*rand()'
        
        # Create inhibitory connections between hemispheres
        S_inhib_left_to_right = Synapses(self.neurons[:N_per_side], self.neurons[N_per_side:], 
                                        'w : 1', on_pre='v_post -= w')
        S_inhib_left_to_right.connect(p=0.05)  # 5% connectivity
        S_inhib_left_to_right.w = '0.4*rand()'
        
        S_inhib_right_to_left = Synapses(self.neurons[N_per_side:], self.neurons[:N_per_side], 
                                        'w : 1', on_pre='v_post -= w')
        S_inhib_right_to_left.connect(p=0.05)  # 5% connectivity
        S_inhib_right_to_left.w = '0.4*rand()'
        
        # Create monitors to record spikes and rates
        self.spike_monitor = SpikeMonitor(self.neurons)
        self.rate_monitor = PopulationRateMonitor(self.neurons)
        
        # Create the network
        self.network = Network(self.neurons, S_left, S_right, 
                              S_inhib_left_to_right, S_inhib_right_to_left,
                              self.spike_monitor, self.rate_monitor)
        
        # Store synapses for later modification
        self.synapses = {
            'left_internal': S_left,
            'right_internal': S_right,
            'left_to_right': S_inhib_left_to_right,
            'right_to_left': S_inhib_right_to_left
        }
        
        print("Brian2 neural network set up successfully")
    
    def connect_to_device(self):
        """Simulate connecting to the MEA device."""
        if not self.device_connected:
            self.setup_network()
            self.device_connected = True
            print("Connected to simulated MEA device")
        return True
    
    def start_recording(self):
        """Start the simulation and recording."""
        if not self.device_connected:
            print("Device not connected. Cannot start recording.")
            return False
        
        print("Recording started")
        return True
    
    def stop_recording(self):
        """Stop the simulation and recording."""
        print("Recording stopped")
    
    def disconnect(self):
        """Disconnect from the simulated MEA device."""
        self.device_connected = False
        print("Disconnected from simulated MEA device")
    
    def run_simulation(self, duration):
        """
        Run the Brian2 simulation for a specified duration.
        
        Parameters:
        -----------
        duration : float
            Duration to run the simulation in milliseconds
        """
        if not self.device_connected:
            print("Device not connected. Cannot run simulation.")
            return
        
        # Run the network
        self.network.run(duration * ms)
        
        # Update the data buffer with the latest spike data
        self._update_buffer()
    
    def _update_buffer(self):
        """Update the data buffer with the latest spike data."""
        # Clear the buffer
        self.data_buffer = np.zeros((self.num_channels, self.buffer_size))
        
        # Get spike data from the last simulation step
        for i in range(self.num_channels):
            # Get spike times for this neuron
            spike_times = self.spike_monitor.t[self.spike_monitor.i == i]
            
            # Convert spike times to indices in the buffer
            if len(spike_times) > 0:
                # Only consider recent spikes that would fit in the buffer
                recent_spikes = spike_times[-self.buffer_size:]
                
                # Convert spike times to buffer indices (normalized to buffer size)
                indices = ((recent_spikes / ms) % self.buffer_size).astype(int)
                
                # Set spikes in the buffer (using a simple spike representation)
                for idx in indices:
                    self.data_buffer[i, idx] = 30  # Represent spike with amplitude of 30
    
    def read_neural_data_buffer(self):
        """Read the current neural data buffer."""
        return self.data_buffer.copy()
    
    def stimulate_neurons(self, pole_angle, pole_angular_velocity, reward):
        """
        Generate and apply stimulation based on pole angle, angular velocity, and reward.
        
        Parameters:
        -----------
        pole_angle : float
            Current angle of the pole
        pole_angular_velocity : float
            Current angular velocity of the pole
        reward : float
            Current reward value
        """
        # Determine which pattern to use based on pole angle
        if np.abs(pole_angle) < 0.262:  # ~15 degrees
            # Positive reward: Generate predictable reinforcing stimulation pattern
            stim_wave = generate_stim_wave(pole_angle, pole_angular_velocity, 100)
            active_group = "reward"
        else:
            # Negative reward or punishment: Generate random noise as unpredictable feedback
            stim_wave = self.generate_random_noise(100)
            active_group = "punishment"
            self.failure_count += 1
        
        # Select channels based on last action
        N_per_side = self.num_channels // 2
        if self.last_action == 0:
            target_neurons = range(N_per_side)  # Left channels
        else:
            target_neurons = range(N_per_side, self.num_channels)  # Right channels
        
        # Convert the stimulation wave to input current for Brian2 neurons
        self._apply_stimulation(stim_wave, target_neurons)
        
        # Run the simulation for a short duration to process the stimulation
        self.run_simulation(100)  # 100 ms of simulation
    
    def _apply_stimulation(self, stim_wave, target_neurons):
        """
        Apply stimulation to the target neurons in the Brian2 network.
        
        Parameters:
        -----------
        stim_wave : numpy.ndarray
            Stimulation waveform
        target_neurons : list or range
            Indices of neurons to stimulate
        """
        # Reset all neuron inputs
        self.neurons.I = 0
        
        # Normalize the stimulation wave to reasonable current values for the model
        # Izhikevich neurons typically use currents in the range of 0-20
        normalized_stim = np.mean(stim_wave) / 10
        
        # Apply the stimulation to target neurons
        for i in target_neurons:
            self.neurons.I[i] = normalized_stim
    
    def extract_neuron_action(self, raw_neural_data, threshold=3):
        """
        Process raw neural data to extract an action (0 or 1) for CartPole.
        
        Parameters:
        -----------
        raw_neural_data : numpy.ndarray
            Raw neural data buffer
        threshold : float
            Threshold for spike detection
            
        Returns:
        --------
        int
            Action (0 or 1) for the CartPole environment
        """
        from spike import MADs, count_spikes
        
        # Create time array for the data
        time_array = np.linspace(0, 1, raw_neural_data.shape[1])
        
        # Calculate median absolute deviations and activity
        median_abs_deviations, abs_activity, _, _ = MADs(time_array, raw_neural_data)
        
        # Count spikes based on threshold
        spike_count = count_spikes(abs_activity, median_abs_deviations, threshold)
        
        num_channels = len(spike_count)
        left_spike_count = np.sum(spike_count[:num_channels // 2])  # First half of channels
        right_spike_count = np.sum(spike_count[num_channels // 2:])  # Second half of channels
        
        action = 0 if left_spike_count > right_spike_count else 1  # 0 for left, 1 for right
        self.last_action = action
        
        return action
    
    def generate_random_noise(self, duration, sampling_rate=500, base_voltage_amp=150):
        """
        Generate a random noise waveform for punishment.
        
        Parameters:
        -----------
        duration : int
            Duration of the noise in milliseconds
        sampling_rate : int
            Sampling rate in Hz
        base_voltage_amp : float
            Base amplitude of the voltage
            
        Returns:
        --------
        numpy.ndarray
            Random noise waveform
        """
        num_samples = int(sampling_rate * (duration / 1000.0))
        t = np.linspace(0, duration / 1000.0, num_samples)
        
        # Increase voltage amplitude based on the number of failures (caps at 2x base amp)
        max_amp_multiplier = min(2, 1 + 0.1 * self.failure_count)
        voltage_amp = base_voltage_amp * max_amp_multiplier
        
        # Introduce frequency randomness based on failure count
        base_freq = 5
        max_freq = min(50, base_freq + 3 * self.failure_count)
        
        # Generate a chaotic oscillation pattern
        random_freq_1 = np.random.uniform(base_freq, max_freq)
        random_freq_2 = np.random.uniform(base_freq * 2, max_freq * 2)
        
        # Generate sine wave components with phase shifts
        noise_wave = (
            voltage_amp * np.sin(2 * np.pi * random_freq_1 * t + np.random.uniform(0, 2*np.pi)) +
            0.5 * voltage_amp * np.sin(2 * np.pi * random_freq_2 * t + np.random.uniform(0, 2*np.pi))
        )
        
        # Add a uniform noise component for extra disruption
        noise_wave += np.random.uniform(-voltage_amp, voltage_amp, num_samples)
        
        return noise_wave
