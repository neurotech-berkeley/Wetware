import socket
import numpy as np
import time
import gym
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time


# Read + Write
NUM_BITS = 8

def read_bytes(numChannels, bufferSize, client_socket, num_bytes_per_element=4):
    result = np.empty(numChannels * bufferSize)

    data = client_socket.recv(bufferSize*numChannels*num_bytes_per_element*NUM_BITS)

    arr = []
    # Receive data from the server
    for i in range(numChannels):
        temp = []
        for j in range(bufferSize):
            temp += [int.from_bytes(data[(i*numChannels*4)+j*4:(i*numChannels*4)+(j+1)*4], "little")]
        arr += [temp]

    # Close the socket
    client_socket.close()

    return result.reshape((numChannels, bufferSize))

def write_bytes():
    pass


# OpenAI Gym
def cartpole(env, agent_action):
    #env = gym.make('CartPole-v1')
    # action_size = 2
    # episodes = 100
    # scores = []
    # state = np.reshape(state, [1, *state_size])
    #done, score = False, 0
    # while not done:
    #     action = generate_action(state)
    #     state, reward, done, _, __ = env.step(action)
    #     send_reward(reward)
    #     score += reward
    #     if done:
    #         scores.append(score)
    #         if not e % 10: 
    #             print(f"episode: {e}/{episodes}, score: {score}")
