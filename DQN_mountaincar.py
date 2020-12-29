# this time solution will be implemented on keras to see the difference
# training process on cpu because I do not have gpu on my local machine
import os
import gym
import keras
import random
import argparse
import collections

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense

parser = argparse.ArgumentParser(description='DQN to solve MountainCar-v0 env')
parser.add_argument('-env', type=str, default='MountainCar-v0',
					help='rl env to solve')
parser.add_argument('-e', type=float, default=1.0, 
					help='epsilon for dqn')
parser.add_argument('-g', type=float, default=0.95, 
					help='gamma for dqn')
parser.add_argument('-b', type=int, default=64, 
					help='batch size')
parser.add_argument('-e_min', type=float, default=0.01, 
					help='minimum value that can be equal args.e')
parser.add_argument('-lr', type=float, default=0.001, 
					help='learning rate')
parser.add_argument('-e_decay', type=float, default=0.995, 
					help='epsilon_decay value')
parser.add_argument('-m', type=int, default=10000, 
					help='memory max capacity')
parser.add_argument('-hd', type=int, default=64, 
					help='hidden_size')
parser.add_argument('-episode', type=int, default=100, 
					help='episodes number')
					
args = parser.parse_args()


class DeepQlearning:
    def __init__(self, action, state):
        self.action_space = action
        self.state_space = state
        self.memory = collections.deque(maxlen=args.m)
        self.model = self.model()

    def model(self):
        model = keras.Sequential()
        model.add(Dense(args.hd, input_dim=self.state_space, activation=keras.activations.relu))
        model.add(Dense(args.hd*2, activation=keras.activations.relu))
        model.add(Dense(self.action_space, activation=keras.activations.linear))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=args.lr))
        return model

    def my_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

	#greedy implementation
    def action(self, state):
        if np.random.rand() <= args.e:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        "replay memory implementation"
        if len(self.memory) < args.b:
            return

        minibatch = random.sample(self.memory, args.b)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + args.g*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(args.b)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if args.e > args.e_min:
            args.e *= args.e_decay


def calc_reward(state):

    if state[0] >= 0.5:
        print("----DONE!----")
        return 10
    if state[0] > -0.4:
        return (1.0+state[0])**2
    return 0


def train_loop(env, episode=args.episode):
    loss = []
    model = DeepQlearning(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 2))
        score = 0
        max_steps = 1000
        for i in range(max_steps):
            action = model.action(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            reward = calc_reward(next_state)
            score += reward
            next_state = np.reshape(next_state, (1, 2))
            model.my_memory(state, action, reward, next_state, done)
            state = next_state
            model.replay()
            if done:
                print(f"episode: {e}/{episode}, score: {score}")
                break
        loss.append(score)
    return loss


def main():
	env = gym.envs.make(args.env)

	states = env.observation_space.shape[0]
	actions = env.action_space.n
	loss = train_loop(env)
	plt.plot([i+1 for i in range(args.episode)], loss)
	plt.show()
	
	
if __name__ == "__main__":
	main()
	