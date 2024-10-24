import gym
from gym.spaces import Discrete
import torch.nn as nn
import torch.optim as optim
import numpy as np
from MCS_Device_Interface import MCS_Device_Interface

class OpenAIGymAPI:
    
    def __init__(self):
        self.env = 'CartPole-v1'
        self.state = None
        self.mcs_interface = MCS_Device_Interface()

    def initialize_training(self):
        self.state = self.env.reset()
        self.total_reward = 0
        return self.state
    
    def run_single_frame(self, raw_data):
        pole_angle = self.state[2]
        pole_angular_velocty = self.state[3]
        self.mcs_interface.send_pole_angle(pole_angle)
        action = self.mcs_interface.receive_neuron_action()
        observation, reward, terminated, _,  _, _, _ = self.env.step(action)
        return observation[2], observation[3], reward, terminated