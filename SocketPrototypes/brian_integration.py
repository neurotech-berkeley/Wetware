# brian_integration.py
import matplotlib.pyplot as plt
from brian_simulator import Brian2MEASimulator
import brian2 as b2
from brian2 import prefs
prefs.codegen.target = "numpy"
import numpy as np
import gymnasium as gym

class Brian2CartPoleIntegration:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.mea_simulator = Brian2MEASimulator(num_channels=60, buffer_size=100)
        self.mea_simulator.connect_to_device()

        self.total_reward = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_length = 0

    def initialize_training(self):
        state, _ = self.env.reset()
        self.total_reward = 0
        self.current_episode_length = 0
        self.mea_simulator.start_recording()
        return state

    def run_single_frame(self):
        self.mea_simulator.run_simulation(10)
        buffer = self.mea_simulator.read_neural_data_buffer()
        action = self.mea_simulator.extract_neuron_action(buffer)
        obs, reward, terminated, truncated, _ = self.env.step(action)

        self.total_reward += reward
        self.current_episode_length += 1

        pole_angle = obs[2]
        pole_angular_velocity = obs[3]
        self.mea_simulator.stimulate_neurons(pole_angle, pole_angular_velocity, reward)

        done = terminated or truncated
        if done:
            self.episode_rewards.append(self.total_reward)
            self.episode_lengths.append(self.current_episode_length)

        return pole_angle, pole_angular_velocity, reward, done

    def train(self, num_episodes=100, render=False):
        for episode in range(num_episodes):
            state = self.initialize_training()
            done = False

            while not done:
                if render:
                    self.env.render()
                _, _, _, done = self.run_single_frame()

            print(f"Episode {episode+1}/{num_episodes}, Reward: {self.total_reward}, Length: {self.current_episode_length}")

        self.env.close()
        self.plot_training_results()

    def plot_training_results(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        plt.tight_layout()
        plt.show()

    def visualize_network_activity(self, duration=500):
        self.mea_simulator.run_simulation(duration)
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.mea_simulator.spike_monitor.t/ms, self.mea_simulator.spike_monitor.i, '.k', markersize=2)
        plt.title('Spike Raster Plot')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')

        plt.subplot(2, 1, 2)
        plt.plot(self.mea_simulator.rate_monitor.t/ms, self.mea_simulator.rate_monitor.rate/Hz)
        plt.title('Population Firing Rate')
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    integration = Brian2CartPoleIntegration()
    integration.train(num_episodes=100, render=False)
    integration.visualize_network_activity(duration=1000)
