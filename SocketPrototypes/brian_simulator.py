# brian_simulator.py (final version with reward trace and correct synapse assignment)
import numpy as np
from brian2 import *
from square import generate_stim_wave

class Brian2MEASimulator:
    def __init__(self, num_channels=60, buffer_size=100):
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        self.data_buffer = np.zeros((self.num_channels, self.buffer_size))
        self.device_connected = False
        self.last_action = 0
        self.failure_count = 0
        self.reward_trace = 0.0  

        defaultclock.dt = 0.1 * ms
        self.setup_network()
        self.device_connected = True

    def setup_network(self):
        N = self.num_channels
        N_half = N // 2

        neuron_eqs = '''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I)/ms : 1
        du/dt = (a*(b*v - u))/ms : 1
        I : 1
        a : 1
        b : 1
        c : 1
        d : 1
        '''

        self.neurons = NeuronGroup(N, neuron_eqs, threshold='v>=30', reset='v=c; u+=d', method='euler')
        self.neurons.a = 0.02
        self.neurons.b = 0.2
        self.neurons.c = -65
        self.neurons.d = 8
        self.neurons.v = -65
        self.neurons.u = self.neurons.b * self.neurons.v

        syn_eqs = '''
        w : 1
        reward : 1
        dApre/dt = -Apre / (20*ms) : 1 (event-driven)
        dApost/dt = -Apost / (20*ms) : 1 (event-driven)
        '''

        on_pre = '''
        Apre += 0.01
        v_post += w
        w = clip(w + reward * Apost, 0, 1.0)
        '''

        on_post = '''
        Apost += -0.012
        w = clip(w + reward * Apre, 0, 1.0)
        '''

        S_left = Synapses(self.neurons[:N_half], self.neurons[:N_half], model=syn_eqs, on_pre=on_pre, on_post=on_post)
        S_right = Synapses(self.neurons[N_half:], self.neurons[N_half:], model=syn_eqs, on_pre=on_pre, on_post=on_post)
        S_left.connect(p=0.1)
        S_right.connect(p=0.1)
        S_left.w = '0.5*rand()'
        S_right.w = '0.5*rand()'

        self.synapses = {'left': S_left, 'right': S_right}
        self.spike_monitor = SpikeMonitor(self.neurons)
        self.rate_monitor = PopulationRateMonitor(self.neurons)

        self.network = Network(self.neurons, S_left, S_right, self.spike_monitor, self.rate_monitor)

    def connect_to_device(self):
        print("Simulated device connected.")

    def start_recording(self):
        print("Started recording (in silico).")

    def run_simulation(self, duration):
        self.network.run(duration * ms)
        self._update_buffer()

    def _update_buffer(self):
        self.data_buffer = np.zeros((self.num_channels, self.buffer_size))
        for i in range(self.num_channels):
            spike_times = self.spike_monitor.t[self.spike_monitor.i == i]
            if len(spike_times) > 0:
                recent_spikes = spike_times[-self.buffer_size:]
                indices = ((recent_spikes / ms) % self.buffer_size).astype(int)
                for idx in indices:
                    self.data_buffer[i, idx] = 30

    def stimulate_neurons(self, pole_angle, pole_angular_velocity, reward):
        if np.abs(pole_angle) < 0.262:
            stim_wave = generate_stim_wave(pole_angle, pole_angular_velocity, 100)
        else:
            stim_wave = self.generate_random_noise(100)
            self.failure_count += 1

        normalized_stim = np.mean(stim_wave) / 10
        self.neurons.I = 0
        N_half = self.num_channels // 2
        target_neurons = range(N_half) if self.last_action == 0 else range(N_half, self.num_channels)
        for i in target_neurons:
            self.neurons.I[i] = normalized_stim

        self.reward_trace = 0.9 * self.reward_trace + 0.1 * reward  # ✅ update reward trace

        for s in [self.synapses['left'], self.synapses['right']]:
            s.reward[:] = self.reward_trace  # ✅ broadcast reward trace

        self.run_simulation(100)

    def extract_neuron_action(self, raw_data, threshold=3):
        from spike import MADs, count_spikes
        time_array = np.linspace(0, 1, raw_data.shape[1])
        med_devs, abs_act, _, _ = MADs(time_array, raw_data)
        spike_count = count_spikes(abs_act, med_devs, threshold)
        left = np.sum(spike_count[:self.num_channels // 2])
        right = np.sum(spike_count[self.num_channels // 2:])
        action = 0 if left > right else 1
        self.last_action = action
        return action

    def read_neural_data_buffer(self):
        return self.data_buffer.copy()

    def generate_random_noise(self, duration, sampling_rate=500, base_voltage_amp=150):
        num_samples = int(sampling_rate * (duration / 1000.0))
        t = np.linspace(0, duration / 1000.0, num_samples)
        voltage_amp = base_voltage_amp * min(2, 1 + 0.1 * self.failure_count)
        base_freq = 5
        max_freq = min(50, base_freq + 3 * self.failure_count)
        freq1 = np.random.uniform(base_freq, max_freq)
        freq2 = np.random.uniform(base_freq*2, max_freq*2)
        noise_wave = (
            voltage_amp * np.sin(2 * np.pi * freq1 * t + np.random.uniform(0, 2*np.pi)) +
            0.5 * voltage_amp * np.sin(2 * np.pi * freq2 * t + np.random.uniform(0, 2*np.pi))
        )
        noise_wave += np.random.uniform(-voltage_amp, voltage_amp, num_samples)
        return noise_wave

