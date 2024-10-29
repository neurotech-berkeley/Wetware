import gym
from gym.spaces import Discrete
import torch.nn as nn
import torch.optim as optim
import numpy as np
from MCS_Device_Interface import MCS_Device_Interface

class OpenAIGymAPI:
    
    def __init__(self, mcs_interface, num_channels, buffer_size):
        self.env = 'CartPole-v1'
        self.state = None
        self.mcs_interface = mcs_interface
        self.num_channels = num_channels
        self.buffer_size = buffer_size

    def initialize_training(self):
        self.state = self.env.reset()
        self.total_reward = 0
        return self.state
    
    def run_single_frame(self, client_socket):
        raw_neural_buffer = self.mcs_interface.read_neural_data_buffer(self.num_channels, self.buffer_size, client_socket)
        neuron_action = self.mcs_interface.extract_neuron_action(raw_neural_buffer)
        
        observation, reward, terminated, _,  _, _, _ = self.env.step(neuron_action)
        
        return observation[2], observation[3], reward, terminated