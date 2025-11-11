#!/usr/bin/env python3
"""
Fixed Brian2 CartPole Simulation
==================================
Uses simulated spiking neural networks to control CartPole via reinforcement learning.
"""

import sys
import os

# Add SocketPrototypes to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SocketPrototypes'))

import numpy as np
import gymnasium as gym
import brian2 as b2
import matplotlib.pyplot as plt

from square import generate_stim_wave
from spike import MADs, count_spikes

# Suppress Brian2 warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')


class VirtualMEA:
    """Virtual MEA that simulates neural recording and stimulation using Brian2."""

    def __init__(self, left_neurons, right_neurons, left_spikes, right_spikes, network):
        self.left_neurons = left_neurons
        self.right_neurons = right_neurons
        self.left_spikes = left_spikes
        self.right_spikes = right_spikes
        self.network = network
        self.last_action = 0
        self.failure_count = 0

    def read_neural_data_buffer(self, num_channels, buffer_size):
        """Simulate reading from MEA by running the Brian2 simulation"""
        # Run simulation for a fixed duration
        self.network.run(20 * b2.ms)

        # Generate simulated voltage traces from spike data
        result = np.zeros((num_channels, buffer_size))

        # Convert spike data to continuous traces for left neurons
        for i, neuron_idx in enumerate(range(len(self.left_neurons))):
            if neuron_idx in self.left_spikes.i:
                spike_times = self.left_spikes.t[self.left_spikes.i == neuron_idx]
                for t in spike_times:
                    # Create spike waveform
                    peak_idx = int((t/b2.ms) * buffer_size/20)
                    if 0 <= peak_idx < buffer_size:
                        result[i, peak_idx:min(peak_idx+5, buffer_size)] = 100

        # Convert spike data for right neurons
        for i, neuron_idx in enumerate(range(len(self.right_neurons))):
            if neuron_idx in self.right_spikes.i:
                spike_times = self.right_spikes.t[self.right_spikes.i == neuron_idx]
                for t in spike_times:
                    peak_idx = int((t/b2.ms) * buffer_size/20)
                    if 0 <= peak_idx < buffer_size:
                        result[i + num_channels//2, peak_idx:min(peak_idx+5, buffer_size)] = 100

        return result

    def stimulate_neurons(self, pole_angle, pole_angular_velocity, reward):
        """Stimulate neurons based on pole angle and angular velocity"""

        # Determine reward vs punishment
        if np.abs(pole_angle) < 0.262:  # ~15 degrees - success
            # Generate predictable stimulation pattern
            duration = 100  # ms
            stim_wave = generate_stim_wave(pole_angle, pole_angular_velocity, duration)
            self.failure_count = 0  # Reset failure count on success
        else:
            # Generate random noise as punishment
            stim_wave = self.generate_random_noise(100)
            self.failure_count += 1

        # Determine which group to stimulate based on angle
        if pole_angle < 0:  # Leaning left
            target_neurons = self.left_neurons
        else:  # Leaning right
            target_neurons = self.right_neurons

        # Convert stimulation wave to current (much stronger to actually trigger spikes)
        # The waveform is in microvolts, we need nanoamps for actual spiking
        stim_current = np.mean(np.abs(stim_wave)) * 0.01 * b2.nA

        # Apply stimulation to target neurons (add random variation across neurons)
        for i in range(len(target_neurons)):
            # Add some variability so not all neurons respond identically
            target_neurons.I[i] = stim_current * (0.8 + 0.4 * np.random.random())

        # Record last action
        self.last_action = 0 if pole_angle < 0 else 1

    def generate_random_noise(self, duration, sampling_rate=500, base_voltage_amp=150):
        """Generate random noise waveform for punishment."""
        num_samples = int(sampling_rate * (duration / 1000.0))
        t = np.linspace(0, duration / 1000.0, num_samples)

        # Increase intensity based on failure count
        max_amp_multiplier = min(2, 1 + 0.1 * self.failure_count)
        voltage_amp = base_voltage_amp * max_amp_multiplier

        # Generate chaotic oscillation
        base_freq = 5
        max_freq = min(50, base_freq + 3 * self.failure_count)

        random_freq_1 = np.random.uniform(base_freq, max_freq)
        random_freq_2 = np.random.uniform(base_freq * 2, max_freq * 2)

        noise_wave = (
            voltage_amp * np.sin(2 * np.pi * random_freq_1 * t + np.random.uniform(0, 2*np.pi)) +
            0.5 * voltage_amp * np.sin(2 * np.pi * random_freq_2 * t + np.random.uniform(0, 2*np.pi))
        )

        noise_wave += np.random.uniform(-voltage_amp/2, voltage_amp/2, num_samples)

        return noise_wave

    def extract_neuron_action(self, raw_neural_data, threshold=3):
        """Process neural data to determine action (0 or 1)"""
        # Create time array for the data
        time_array = np.linspace(0, 1, raw_neural_data.shape[1])

        # Calculate median absolute deviations and activity
        median_abs_deviations, abs_activity, _, _ = MADs(time_array, raw_neural_data)

        # Count spikes based on threshold
        spike_count = count_spikes(abs_activity, median_abs_deviations, threshold)

        num_channels = len(spike_count)
        left_spike_count = np.sum(spike_count[:num_channels // 2])
        right_spike_count = np.sum(spike_count[num_channels // 2:])

        # Determine action
        action = 0 if left_spike_count > right_spike_count else 1
        self.last_action = action

        return action


class OpenAIGymAPI:
    """Interface between Virtual MEA and OpenAI Gym environment."""

    def __init__(self, virtual_mea, num_channels, buffer_size):
        self.env = gym.make('CartPole-v1')
        self.virtual_mea = virtual_mea
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        self.state = None
        self.total_reward = 0

    def initialize_training(self):
        """Initialize training by resetting the environment."""
        self.state, _ = self.env.reset()
        self.total_reward = 0
        return self.state

    def run_single_frame(self):
        """Run a single frame of the CartPole environment."""
        # Get neural data from the virtual MEA
        raw_neural_buffer = self.virtual_mea.read_neural_data_buffer(
            self.num_channels, self.buffer_size
        )

        # Extract action from neural activity
        neuron_action = self.virtual_mea.extract_neuron_action(raw_neural_buffer)

        # Take a step in the environment
        observation, reward, terminated, truncated, info = self.env.step(neuron_action)

        # Update state and total reward
        self.state = observation
        self.total_reward += reward

        # Return pole angle, angular velocity, reward, and done flag
        pole_angle = observation[2]
        pole_angular_velocity = observation[3]
        done = terminated or truncated

        return pole_angle, pole_angular_velocity, reward, done


def setup_brian2_network():
    """Set up the Brian2 spiking neural network."""

    print("Setting up Brian2 neural network...")

    # Start Brian2 simulation scope
    b2.start_scope()

    # Network parameters
    num_channels = 60
    simulation_dt = 0.1 * b2.ms

    # Neuron model parameters (Adaptive Exponential Integrate-and-Fire)
    EL = -70 * b2.mV      # Resting potential
    VT = -50 * b2.mV      # Spike threshold
    Delta_T = 2 * b2.mV   # Slope factor
    gL = 10 * b2.nsiemens # Leak conductance
    C = 200 * b2.pfarad   # Membrane capacitance
    a = 2 * b2.nsiemens   # Adaptation conductance
    tau_w = 100 * b2.ms   # Adaptation time constant
    b_adapt = 0.05 * b2.nA # Adaptation increment

    namespace = {
        'EL': EL, 'VT': VT, 'Delta_T': Delta_T,
        'gL': gL, 'C': C, 'a': a, 'tau_w': tau_w,
        'b': b_adapt
    }

    # Define neuron equations
    neuron_eqs = '''
    dv/dt = (gL*(EL-v) + gL*Delta_T*exp((v-VT)/Delta_T) - w + I)/C : volt
    dw/dt = (a*(v-EL) - w)/tau_w : amp
    I : amp
    '''

    # Create left and right neuron groups (30 each for 60 total)
    left_neurons = b2.NeuronGroup(
        num_channels // 2, neuron_eqs,
        threshold='v > -50*mV',
        reset='v = -70*mV; w += b',
        method='euler',
        namespace=namespace
    )

    right_neurons = b2.NeuronGroup(
        num_channels // 2, neuron_eqs,
        threshold='v > -50*mV',
        reset='v = -70*mV; w += b',
        method='euler',
        namespace=namespace
    )

    # Initialize neuron states
    for neurons in [left_neurons, right_neurons]:
        neurons.v = EL
        neurons.w = 0 * b2.pA

    # Set up spike monitors
    left_spikes = b2.SpikeMonitor(left_neurons)
    right_spikes = b2.SpikeMonitor(right_neurons)

    # Build network
    network = b2.Network(left_neurons, right_neurons, left_spikes, right_spikes)

    print(f"✓ Created {num_channels} neurons ({num_channels//2} left, {num_channels//2} right)")

    return left_neurons, right_neurons, left_spikes, right_spikes, network, num_channels


def run_simulation(episodes=10, visualize=False):
    """Run the DishBrain simulation."""

    print("\n" + "="*60)
    print("WETWARE DISHBRAIN SIMULATION - Brian2 + CartPole")
    print("="*60 + "\n")

    # Set up neural network
    left_neurons, right_neurons, left_spikes, right_spikes, network, num_channels = setup_brian2_network()

    # Create virtual MEA
    buffer_size = 100
    virtual_mea = VirtualMEA(left_neurons, right_neurons, left_spikes, right_spikes, network)

    # Create OpenAI Gym API
    gym_api = OpenAIGymAPI(virtual_mea, num_channels, buffer_size)

    # Storage for results
    episode_rewards = []
    episode_steps = []

    print(f"\nRunning {episodes} episodes...\n")

    # Run episodes
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")

        # Reset environment
        gym_api.initialize_training()
        done = False
        step_count = 0

        # Run until episode is done
        while not done:
            # Run one frame
            pole_angle, pole_angular_velocity, reward, terminated = gym_api.run_single_frame()

            # Stimulate neurons based on state and reward
            virtual_mea.stimulate_neurons(pole_angle, pole_angular_velocity, reward)

            done = terminated
            step_count += 1

            # Print progress every 50 steps
            if step_count % 50 == 0:
                print(f"  Step {step_count}... (reward so far: {gym_api.total_reward:.0f})")

            # Safety limit
            if step_count > 500:
                print(f"  Reached safety limit of 500 steps!")
                break

        episode_rewards.append(gym_api.total_reward)
        episode_steps.append(step_count)

        print(f"✓ Episode {episode + 1} completed: {step_count} steps, {gym_api.total_reward:.1f} reward\n")

    # Print summary
    print("="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average steps:  {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")
    print(f"Best episode:   {np.max(episode_rewards):.1f} reward in {episode_steps[np.argmax(episode_rewards)]} steps")
    print(f"Worst episode:  {np.min(episode_rewards):.1f} reward in {episode_steps[np.argmin(episode_rewards)]} steps")

    # Plot results
    if visualize:
        plot_results(episode_rewards, episode_steps, episodes)

    return episode_rewards, episode_steps


def plot_results(episode_rewards, episode_steps, episodes):
    """Plot learning progress."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot rewards
    ax1.plot(range(1, episodes+1), episode_rewards, marker='o', linewidth=2)
    ax1.axhline(y=np.mean(episode_rewards), color='r', linestyle='--',
                label=f'Mean: {np.mean(episode_rewards):.1f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Learning Progress: Reward per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot steps
    ax2.plot(range(1, episodes+1), episode_steps, marker='s',
             linewidth=2, color='orange')
    ax2.axhline(y=np.mean(episode_steps), color='r', linestyle='--',
                label=f'Mean: {np.mean(episode_steps):.1f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps Survived')
    ax2.set_title('Survival Time per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('brian2_cartpole_results.png', dpi=150)
    print(f"\n✓ Results saved to 'brian2_cartpole_results.png'")
    plt.show()


if __name__ == "__main__":
    # Run simulation with 10 episodes by default
    run_simulation(episodes=10, visualize=True)
