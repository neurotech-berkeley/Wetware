import gym
import numpy as np

class IntegratedOpenAIGymAPI:
    def __init__(self, mea_interface):
        """Initialize the OpenAI Gym API with the integrated MEA interface."""
        self.env = gym.make('CartPole-v1')
        self.mea_interface = mea_interface
        self.total_reward = 0
    
    def initialize_training(self):
        """Initialize the training environment."""
        state, _ = self.env.reset()
        self.total_reward = 0
        return state
    
    def run_single_frame(self):
        """Run a single frame of the CartPole environment."""
        # Get neural data from the MEA
        raw_neural_buffer = self.mea_interface.read_neural_data_buffer()
        
        # Extract action from neural activity
        neuron_action = self.mea_interface.extract_neuron_action(raw_neural_buffer)
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = self.env.step(neuron_action)
        
        # Update total reward
        self.total_reward += reward
        
        # Return relevant information
        return observation[2], observation[3], reward, terminated or truncated
