#!/usr/bin/env python3
"""
Enhanced Brian2 CartPole Simulation with Comprehensive Neural Visualization
============================================================================
Multiple visualization modes for neural activity analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SocketPrototypes'))

import numpy as np
import gymnasium as gym
import brian2 as b2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict

from square import generate_stim_wave
from spike import MADs, count_spikes

import warnings
warnings.filterwarnings('ignore')


class NeuralActivityTracker:
    """Tracks detailed neural activity for visualization."""

    def __init__(self):
        self.spike_times_left = []
        self.spike_times_right = []
        self.spike_neurons_left = []
        self.spike_neurons_right = []
        self.firing_rates_left = []
        self.firing_rates_right = []
        self.actions_taken = []
        self.rewards_received = []
        self.pole_angles = []
        self.stimulation_currents = []
        self.episode_boundaries = []
        self.current_time = 0

    def record_spikes(self, left_spikes, right_spikes, episode_step):
        """Record spike data from monitors."""
        # Store spike times and neuron indices
        if len(left_spikes.t) > 0:
            self.spike_times_left.extend(left_spikes.t/b2.ms + self.current_time)
            self.spike_neurons_left.extend(left_spikes.i[:])

        if len(right_spikes.t) > 0:
            self.spike_times_right.extend(right_spikes.t/b2.ms + self.current_time)
            self.spike_neurons_right.extend(right_spikes.i[:])

    def record_frame(self, action, reward, pole_angle, stim_current, left_count, right_count):
        """Record frame-level data."""
        self.actions_taken.append(action)
        self.rewards_received.append(reward)
        self.pole_angles.append(pole_angle)
        self.stimulation_currents.append(stim_current)
        self.firing_rates_left.append(left_count)
        self.firing_rates_right.append(right_count)
        self.current_time += 20  # 20ms per frame

    def record_episode_boundary(self):
        """Mark episode boundaries."""
        self.episode_boundaries.append(len(self.actions_taken))


class VirtualMEA:
    """Virtual MEA with enhanced tracking."""

    def __init__(self, left_neurons, right_neurons, left_spikes, right_spikes, network, tracker):
        self.left_neurons = left_neurons
        self.right_neurons = right_neurons
        self.left_spikes = left_spikes
        self.right_spikes = right_spikes
        self.network = network
        self.tracker = tracker
        self.last_action = 0
        self.failure_count = 0
        self.last_stim_current = 0

    def read_neural_data_buffer(self, num_channels, buffer_size):
        """Run simulation and record activity."""
        # Record spikes before clearing
        self.tracker.record_spikes(self.left_spikes, self.right_spikes, 0)

        # Run simulation
        self.network.run(20 * b2.ms)

        # Generate voltage traces
        result = np.zeros((num_channels, buffer_size))

        for i, neuron_idx in enumerate(range(len(self.left_neurons))):
            if neuron_idx in self.left_spikes.i:
                spike_times = self.left_spikes.t[self.left_spikes.i == neuron_idx]
                for t in spike_times:
                    peak_idx = int((t/b2.ms) * buffer_size/20)
                    if 0 <= peak_idx < buffer_size:
                        result[i, peak_idx:min(peak_idx+5, buffer_size)] = 100

        for i, neuron_idx in enumerate(range(len(self.right_neurons))):
            if neuron_idx in self.right_spikes.i:
                spike_times = self.right_spikes.t[self.right_spikes.i == neuron_idx]
                for t in spike_times:
                    peak_idx = int((t/b2.ms) * buffer_size/20)
                    if 0 <= peak_idx < buffer_size:
                        result[i + num_channels//2, peak_idx:min(peak_idx+5, buffer_size)] = 100

        return result

    def stimulate_neurons(self, pole_angle, pole_angular_velocity, reward):
        """Stimulate neurons and track current."""
        if np.abs(pole_angle) < 0.262:
            duration = 100
            stim_wave = generate_stim_wave(pole_angle, pole_angular_velocity, duration)
            self.failure_count = 0
        else:
            stim_wave = self.generate_random_noise(100)
            self.failure_count += 1

        if pole_angle < 0:
            target_neurons = self.left_neurons
        else:
            target_neurons = self.right_neurons

        stim_current = np.mean(np.abs(stim_wave)) * 0.01
        self.last_stim_current = stim_current

        for i in range(len(target_neurons)):
            target_neurons.I[i] = stim_current * (0.8 + 0.4 * np.random.random()) * b2.nA

        self.last_action = 0 if pole_angle < 0 else 1

    def generate_random_noise(self, duration, sampling_rate=500, base_voltage_amp=150):
        """Generate random noise for punishment."""
        num_samples = int(sampling_rate * (duration / 1000.0))
        t = np.linspace(0, duration / 1000.0, num_samples)

        max_amp_multiplier = min(2, 1 + 0.1 * self.failure_count)
        voltage_amp = base_voltage_amp * max_amp_multiplier

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
        """Extract action and record firing rates."""
        time_array = np.linspace(0, 1, raw_neural_data.shape[1])
        median_abs_deviations, abs_activity, _, _ = MADs(time_array, raw_neural_data)
        spike_count = count_spikes(abs_activity, median_abs_deviations, threshold)

        num_channels = len(spike_count)
        left_spike_count = np.sum(spike_count[:num_channels // 2])
        right_spike_count = np.sum(spike_count[num_channels // 2:])

        action = 0 if left_spike_count > right_spike_count else 1
        self.last_action = action

        return action, left_spike_count, right_spike_count


class OpenAIGymAPI:
    """Gym interface with tracking."""

    def __init__(self, virtual_mea, num_channels, buffer_size):
        self.env = gym.make('CartPole-v1')
        self.virtual_mea = virtual_mea
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        self.state = None
        self.total_reward = 0

    def initialize_training(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0
        return self.state

    def run_single_frame(self):
        raw_neural_buffer = self.virtual_mea.read_neural_data_buffer(
            self.num_channels, self.buffer_size
        )

        neuron_action, left_count, right_count = self.virtual_mea.extract_neuron_action(raw_neural_buffer)

        observation, reward, terminated, truncated, info = self.env.step(neuron_action)

        self.state = observation
        self.total_reward += reward

        pole_angle = observation[2]
        pole_angular_velocity = observation[3]
        done = terminated or truncated

        # Record to tracker
        self.virtual_mea.tracker.record_frame(
            neuron_action, reward, pole_angle,
            self.virtual_mea.last_stim_current,
            left_count, right_count
        )

        return pole_angle, pole_angular_velocity, reward, done


def setup_brian2_network(tracker):
    """Set up Brian2 network with tracking."""
    print("Setting up Brian2 neural network...")

    b2.start_scope()

    num_channels = 60
    EL = -70 * b2.mV
    VT = -50 * b2.mV
    Delta_T = 2 * b2.mV
    gL = 10 * b2.nsiemens
    C = 200 * b2.pfarad
    a = 2 * b2.nsiemens
    tau_w = 100 * b2.ms
    b_adapt = 0.05 * b2.nA

    namespace = {
        'EL': EL, 'VT': VT, 'Delta_T': Delta_T,
        'gL': gL, 'C': C, 'a': a, 'tau_w': tau_w,
        'b': b_adapt
    }

    neuron_eqs = '''
    dv/dt = (gL*(EL-v) + gL*Delta_T*exp((v-VT)/Delta_T) - w + I)/C : volt
    dw/dt = (a*(v-EL) - w)/tau_w : amp
    I : amp
    '''

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

    for neurons in [left_neurons, right_neurons]:
        neurons.v = EL
        neurons.w = 0 * b2.pA

    left_spikes = b2.SpikeMonitor(left_neurons)
    right_spikes = b2.SpikeMonitor(right_neurons)

    network = b2.Network(left_neurons, right_neurons, left_spikes, right_spikes)

    print(f"✓ Created {num_channels} neurons ({num_channels//2} left, {num_channels//2} right)")

    return left_neurons, right_neurons, left_spikes, right_spikes, network, num_channels


def run_simulation(episodes=50):
    """Run simulation with comprehensive tracking."""

    print("\n" + "="*60)
    print("ENHANCED WETWARE DISHBRAIN SIMULATION")
    print("="*60 + "\n")

    # Create tracker
    tracker = NeuralActivityTracker()

    # Setup network
    left_neurons, right_neurons, left_spikes, right_spikes, network, num_channels = setup_brian2_network(tracker)

    buffer_size = 100
    virtual_mea = VirtualMEA(left_neurons, right_neurons, left_spikes, right_spikes, network, tracker)
    gym_api = OpenAIGymAPI(virtual_mea, num_channels, buffer_size)

    episode_rewards = []
    episode_steps = []

    print(f"Running {episodes} episodes...\n")

    for episode in range(episodes):
        if episode % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}")

        gym_api.initialize_training()
        done = False
        step_count = 0

        while not done:
            pole_angle, pole_angular_velocity, reward, terminated = gym_api.run_single_frame()
            virtual_mea.stimulate_neurons(pole_angle, pole_angular_velocity, reward)

            done = terminated
            step_count += 1

            if step_count > 500:
                break

        episode_rewards.append(gym_api.total_reward)
        episode_steps.append(step_count)
        tracker.record_episode_boundary()

        if episode % 10 == 0:
            print(f"  → {step_count} steps, {gym_api.total_reward:.0f} reward\n")

    # Print summary
    print("="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Episodes:       {episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average steps:  {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")
    print(f"Best episode:   {np.max(episode_rewards):.1f} reward in {episode_steps[np.argmax(episode_rewards)]} steps")
    print(f"Worst episode:  {np.min(episode_rewards):.1f} reward in {episode_steps[np.argmin(episode_rewards)]} steps")
    print(f"Median reward:  {np.median(episode_rewards):.1f}")
    print(f"\nTotal spikes recorded:")
    print(f"  Left neurons:  {len(tracker.spike_times_left)}")
    print(f"  Right neurons: {len(tracker.spike_times_right)}")
    print(f"\nGenerating visualizations...")

    # Create all visualizations
    create_comprehensive_visualizations(tracker, episode_rewards, episode_steps, episodes)

    return episode_rewards, episode_steps, tracker


def create_comprehensive_visualizations(tracker, episode_rewards, episode_steps, episodes):
    """Create multiple visualization panels."""

    # Figure 1: Performance Overview
    fig1 = plt.figure(figsize=(16, 10))
    gs1 = GridSpec(3, 3, figure=fig1, hspace=0.3, wspace=0.3)

    # Performance over episodes
    ax1 = fig1.add_subplot(gs1[0, :])
    ax1.plot(range(1, episodes+1), episode_rewards, linewidth=2, alpha=0.6, label='Reward')
    ax1.plot(range(1, episodes+1), episode_steps, linewidth=2, alpha=0.6, label='Steps')

    # Add moving average
    window = min(10, episodes//5)
    if window > 1:
        reward_ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        steps_ma = np.convolve(episode_steps, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, episodes+1), reward_ma, linewidth=3, label=f'Reward MA({window})', color='blue')
        ax1.plot(range(window, episodes+1), steps_ma, linewidth=3, label=f'Steps MA({window})', color='orange')

    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Learning Progress Over Episodes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reward distribution
    ax2 = fig1.add_subplot(gs1[1, 0])
    ax2.hist(episode_rewards, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(episode_rewards), color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.axvline(np.median(episode_rewards), color='green', linestyle='--', linewidth=2, label='Median')
    ax2.set_xlabel('Reward', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Action distribution
    ax3 = fig1.add_subplot(gs1[1, 1])
    actions = np.array(tracker.actions_taken)
    action_counts = [np.sum(actions == 0), np.sum(actions == 1)]
    ax3.bar(['Left (0)', 'Right (1)'], action_counts, color=['blue', 'orange'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('Action Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Firing rate comparison
    ax4 = fig1.add_subplot(gs1[1, 2])
    left_rates = np.array(tracker.firing_rates_left)
    right_rates = np.array(tracker.firing_rates_right)
    ax4.boxplot([left_rates, right_rates], labels=['Left', 'Right'])
    ax4.set_ylabel('Spike Count', fontsize=10)
    ax4.set_title('Firing Rate Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Pole angle over time (sample)
    ax5 = fig1.add_subplot(gs1[2, 0])
    sample_frames = min(500, len(tracker.pole_angles))
    ax5.plot(tracker.pole_angles[:sample_frames], linewidth=1)
    ax5.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Frame', fontsize=10)
    ax5.set_ylabel('Pole Angle (rad)', fontsize=10)
    ax5.set_title('Pole Angle (First 500 frames)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Stimulation current over time (sample)
    ax6 = fig1.add_subplot(gs1[2, 1])
    ax6.plot(tracker.stimulation_currents[:sample_frames], linewidth=1, color='purple')
    ax6.set_xlabel('Frame', fontsize=10)
    ax6.set_ylabel('Stim Current', fontsize=10)
    ax6.set_title('Stimulation Current (First 500 frames)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Firing rates over time (sample)
    ax7 = fig1.add_subplot(gs1[2, 2])
    ax7.plot(tracker.firing_rates_left[:sample_frames], label='Left', alpha=0.7, linewidth=1)
    ax7.plot(tracker.firing_rates_right[:sample_frames], label='Right', alpha=0.7, linewidth=1)
    ax7.set_xlabel('Frame', fontsize=10)
    ax7.set_ylabel('Spike Count', fontsize=10)
    ax7.set_title('Neural Activity (First 500 frames)', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    plt.suptitle('Wetware DishBrain - Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('neural_analysis_performance.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: neural_analysis_performance.png")

    # Figure 2: Spike Raster Plot
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Left neurons spike raster
    if len(tracker.spike_times_left) > 0:
        ax1.scatter(tracker.spike_times_left, tracker.spike_neurons_left,
                   s=1, alpha=0.6, color='blue')
    ax1.set_ylabel('Left Neuron Index', fontsize=12)
    ax1.set_title('Left Hemisphere Neural Activity (Spike Raster)', fontsize=14, fontweight='bold')
    ax1.set_ylim(-1, 30)
    ax1.grid(True, alpha=0.3, axis='x')

    # Right neurons spike raster
    if len(tracker.spike_times_right) > 0:
        ax2.scatter(tracker.spike_times_right, tracker.spike_neurons_right,
                   s=1, alpha=0.6, color='orange')
    ax2.set_ylabel('Right Neuron Index', fontsize=12)
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_title('Right Hemisphere Neural Activity (Spike Raster)', fontsize=14, fontweight='bold')
    ax2.set_ylim(-1, 30)
    ax2.grid(True, alpha=0.3, axis='x')

    # Add episode boundaries
    for boundary_idx in tracker.episode_boundaries[:10]:  # First 10 episodes
        if boundary_idx < len(tracker.actions_taken):
            boundary_time = boundary_idx * 20
            ax1.axvline(boundary_time, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax2.axvline(boundary_time, color='red', linestyle='--', alpha=0.3, linewidth=1)

    plt.suptitle('Wetware DishBrain - Spike Raster Plot', fontsize=16, fontweight='bold')
    plt.savefig('neural_analysis_raster.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: neural_analysis_raster.png")

    # Figure 3: Neural Activity Heatmap
    fig3, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Create activity heatmaps
    sample_size = min(1000, len(tracker.firing_rates_left))

    # Left firing rates over time
    left_matrix = np.array(tracker.firing_rates_left[:sample_size]).reshape(-1, 1).T
    im1 = axes[0, 0].imshow(left_matrix, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0, 0].set_ylabel('Left Population', fontsize=10)
    axes[0, 0].set_title('Left Neuron Activity Over Time', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0], label='Spike Count')

    # Right firing rates over time
    right_matrix = np.array(tracker.firing_rates_right[:sample_size]).reshape(-1, 1).T
    im2 = axes[0, 1].imshow(right_matrix, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0, 1].set_ylabel('Right Population', fontsize=10)
    axes[0, 1].set_title('Right Neuron Activity Over Time', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1], label='Spike Count')

    # Action-conditioned firing rates
    actions_sample = np.array(tracker.actions_taken[:sample_size])
    left_when_left = [l for l, a in zip(tracker.firing_rates_left[:sample_size], actions_sample) if a == 0]
    left_when_right = [l for l, a in zip(tracker.firing_rates_left[:sample_size], actions_sample) if a == 1]
    right_when_left = [r for r, a in zip(tracker.firing_rates_right[:sample_size], actions_sample) if a == 0]
    right_when_right = [r for r, a in zip(tracker.firing_rates_right[:sample_size], actions_sample) if a == 1]

    axes[1, 0].hist([left_when_left, left_when_right], label=['Action Left', 'Action Right'],
                    bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Spike Count', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].set_title('Left Neuron Activity by Action', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist([right_when_left, right_when_right], label=['Action Left', 'Action Right'],
                    bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Spike Count', fontsize=10)
    axes[1, 1].set_ylabel('Frequency', fontsize=10)
    axes[1, 1].set_title('Right Neuron Activity by Action', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Wetware DishBrain - Neural Activity Heatmaps', fontsize=16, fontweight='bold')
    plt.savefig('neural_analysis_heatmaps.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: neural_analysis_heatmaps.png")

    plt.close('all')
    print("\n✓ All visualizations completed!")


if __name__ == "__main__":
    # Run extended simulation with comprehensive tracking
    run_simulation(episodes=50)
