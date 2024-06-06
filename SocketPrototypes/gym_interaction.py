import numpy as np
import gym
from spike import *
from scipy.special import softmax

#CURRENTLY WITH DUMMY DATA


env = gym.make('CartPole-v1')
def dummy_gym_play():
    env = gym.make('CartPole-v1')
    action_size = 2
    episodes = 100
    scores = []
    for e in range(episodes):
        state, _ = env.reset()
        # state = np.reshape(state, [1, *state_size])
        done, score = False, 0
        counter = 0
        while not done:
            counter += 1
            # action = generate_action(state)
            #keep pushing cart left
            # for now i'm going to make a distribution out of the first 2 dummy channels
            # we need to come up with a better aggregation scheme
            logits = get_dummy_counts()[:2]
            probs = softmax(logits)
            action = np.argmax(probs)
            state, reward, done, _, __ = env.step(action)
            if np.abs(state[2]) < 12 and done:
                print("terminating because of angle")
            if counter % 100 == 0:
                print(state[0])
            score += reward
            if done:
                scores.append(score)
                if not e % 10:
                    print(f"episode: {e}/{episodes}, score: {score}")

dummy_gym_play()