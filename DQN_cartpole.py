# Here is implemented DQN with replay memory because without replay memory we were not able to reach reward = 200

import os
import gym
import copy
import time
import torch
import random
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable


parser = argparse.ArgumentParser(description='DQN with replay memory for CartPole-v0 env')
parser.add_argument('-env', type=str, default='CartPole-v0',
					help='rl env to solve')
parser.add_argument('-t', type=str, default='DQN with replay memory reward curve', 
					help='title for plot')
parser.add_argument('-img', type=str, default='img', 
					help='path to save graphics')
parser.add_argument('-hd', type=int, default=64, 
					help='hidden_size for nn')
parser.add_argument('-lr', type=float, default=0.05, 
					help='learning rate for nn')
parser.add_argument('-g', type=float, default=0.9, 
					help='gamma in qlearing function')
parser.add_argument('-eps', type=float, default=0.3, 
					help='epsilon in qlearning function')
parser.add_argument('-eps_decay', type=float, default=0.99, 
					help='epsilon decay in qlearning function')
parser.add_argument('-replay', type=bool, default=True, 
					help='if True using DQN with replay memory if false simple DQN')
parser.add_argument('-replay_size', type=int, default=20, 
					help='replay size in qlearning function')
parser.add_argument('-episodes', type=int, default=200, 
					help='episodes to run')
					
					
args = parser.parse_args()


# plot reward curve
def reward_curve(values, title=args.t):   
    f, ax = plt.subplots(figsize=(15,8))
    f.suptitle(title)
    ax.plot(values, label='score per run')
    ax.axhline(195, c='red',ls='-', label='goal')
    ax.set_xlabel('episodes')
    ax.set_ylabel('reward')
    x = range(len(values))
    ax.legend()
    ax.grid()
    plt.savefig(os.path.join(args.img, 'reward_curve.png'))


def q_learning(env, model, episodes=args.episodes, gamma=args.g, epsilon=args.eps, eps_decay=args.eps_decay, replay=args.replay, replay_size=args.replay_size):
    res = []
    memory = []
    episode_i=0
    sum_total_replay_time=0
    for episode in range(episodes):
        episode_i+=1
        
        # Reset state
        state = env.reset()
        done = False
        total = 0
        
        while not done:
            # greedy search to find state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()
            
            # add reward to total reward
            next_state, reward, done, _ = env.step(action)
            
            # Update total and memory
            total += reward
            memory.append((state, action, next_state, reward, done))
            q_values = model.predict(state).tolist()
             
            if done:
                if not replay:
                    q_values[action] = reward
                    model.update(state, q_values)
                break

            if replay:
                t0=time.time()
                model.replay_memory(memory, replay_size, gamma)
                t1=time.time()
                sum_total_replay_time+=(t1-t0)
            else: 
                q_values_next = model.predict(next_state)
                q_values[action] = reward + gamma * torch.max(q_values_next).item()
                model.update(state, q_values)

            state = next_state
        
        epsilon = max(epsilon * eps_decay, 0.01)
        res.append(total)
        reward_curve(res)
        print(f'episode: {episode_i}, total_reward: {total}')
        if replay:
            print("Average replay time:", sum_total_replay_time/episode_i)
        
    return res
	
	
# pytorch realization of simple NN
class DeepQlearning():
    def __init__(self, state_dim, action_dim, hidden_size=args.hd, lr=args.lr):
            self.criterion = torch.nn.MSELoss()
            self.model = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, hidden_size),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden_size, hidden_size*2),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden_size*2, action_dim)
                    )
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr) #seems results are better with SGD rather then with Adam



    def update(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, state):
	# compute q-values
        with torch.no_grad():
            return self.model(torch.Tensor(state))
			

# in fact we inherit "DeepQlearning" class and expand it with replay method
class DeepQlearning_replay_mode(DeepQlearning):
    def replay_memory(self, memory, size, gamma=args.g):
        if len(memory) >= size:
            batch = random.sample(memory,size)
            batch_t = list(map(list, zip(*batch))) #Transpose batch list
            states = batch_t[0]
            actions = batch_t[1]
            next_states = batch_t[2]
            rewards = batch_t[3]
            is_dones = batch_t[4]
        
            states = torch.Tensor(states)
            actions_tensor = torch.Tensor(actions)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
            is_dones_indices = torch.where(is_dones_tensor==True)[0]
        
            all_q_values = self.model(states)
            all_q_values_next = self.model(next_states)
            all_q_values[range(len(all_q_values)),actions]=rewards+gamma*torch.max(all_q_values_next, axis=1).values
            all_q_values[is_dones_indices.tolist(), actions_tensor[is_dones].tolist()]=rewards[is_dones_indices.tolist()] 
            self.update(states.tolist(), all_q_values.tolist())


def main():
	# declare env
	env = gym.envs.make(args.env)
	states = env.observation_space.shape[0]
	actions = env.action_space.n
	
	#decide to run simple dqn or dqn with replay_memory
	if args.replay:
		dqn_replay_memory = DeepQlearning_replay_mode(states, actions)
		replay_memory = q_learning(env, dqn_replay_memory)
	else:
		simple_dqn = DQN(states, actions)
		simple = q_learning(env, simple_dqn)
	print('------- DONE! -------')

	
if __name__ == '__main__':
	main()
