# brian2_cartpole_integration.py
import gym
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from brian_simulator import Brian2MEASimulator

class Brian2CartPoleIntegration:
    """
    A class that integrates a Brian2 spiking neural network with the CartPole environment.
    """
    
    def __init__(self):
        """Initialize the integration between Brian2 and CartPole."""
        # Create the CartPole environment
        self.env = gym.make('CartPole-v1')
        
        # Create the Brian2 MEA simulator
        self.mea_simulator = Brian2MEASimulator(num_channels=60, buffer_size=100)
        
        # Connect to the simulated device
        self.mea_simulator.connect_to_device()
        
        # Training metrics
        self.total_reward = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_length = 0
    
    def initialize_training(self):
        """Initialize the training environment."""
        state, _ = self.env.reset()
        self.total_reward = 0
        self.current_episode_length = 0
        
        # Start recording from the simulated MEA
        self.mea_simulator.start_recording()
        
        return state
    
    def run_single_frame(self):
        """Run a single frame of the CartPole environment."""
        # Run the Brian2 simulation for a short duration
        self.mea_simulator.run_simulation(10)  # 10 ms of simulation
        
        # Get neural data from the MEA simulator
        raw_neural_buffer = self.mea_simulator.read_neural_data_buffer()
        
        # Extract action from neural activity
        neuron_action = self.mea_simulator.extract_neuron_action(raw_neural_buffer)
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = self.env.step(neuron_action)
        
        # Update total reward and episode length
        self.total_reward += reward
        self.current_episode_length += 1
        
        # Provide feedback to the neural network based on the environment state
        pole_angle = observation[2]
        pole_angular_velocity = observation[3]
        
        # Stimulate the neurons based on the current state
        self.mea_simulator.stimulate_neurons(pole_angle, pole_angular_velocity, reward)
        
        # Check if episode is done
        done = terminated or truncated
        if done:
            self.episode_rewards.append(self.total_reward)
            self.episode_lengths.append(self.current_episode_length)
        
        # Return relevant information
        return pole_angle, pole_angular_velocity, reward, done
    
    def train(self, num_episodes=100, render=False):
        """
        Train the Brian2 neural network to play CartPole.
        
        Parameters:
        -----------
        num_episodes : int
            Number of episodes to train for
        render : bool
            Whether to render the environment
        """
        for episode in range(num_episodes):
            # Initialize the environment
            state = self.initialize_training()
            done = False
            
            # Run the episode
            while not done:
                if render:
                    self.env.render()
                
                # Run a single frame
                pole_angle, pole_angular_velocity, reward, done = self.run_single_frame()
            
            # Print episode results
            print(f"Episode {episode+1}/{num_episodes}, Reward: {self.total_reward}, Length: {self.current_episode_length}")
        
        # Close the environment
        self.env.close()
        
        # Plot training results
        self.plot_training_results()
    
    def plot_training_results(self):
        """Plot the training results."""
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
        """
        Visualize the activity of the Brian2 network.
        
        Parameters:
        -----------
        duration : int
            Duration to run the simulation in milliseconds
        """
        # Run the simulation
        self.mea_simulator.run_simulation(duration)
        
        # Plot the results
        plt.figure(figsize=(12, 8))
        
        # Plot spike raster
        plt.subplot(2, 1, 1)
        plt.plot(self.mea_simulator.spike_monitor.t/ms, self.mea_simulator.spike_monitor.i, '.k', markersize=2)
        plt.title('Spike Raster Plot')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        
        # Plot population rate
        plt.subplot(2, 1, 2)
        plt.plot(self.mea_simulator.rate_monitor.t/ms, self.mea_simulator.rate_monitor.rate/Hz)
        plt.title('Population Firing Rate')
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')
        
        plt.tight_layout()
        plt.show()
