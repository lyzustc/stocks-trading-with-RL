import numpy as np
import torch
import gym.wrappers
from dqn_agent import DQNAgent
from environ import StocksEnv
from collections import deque

class DQNAlgo(object):
    def __init__(self, filepath, netname, algoname, eps_start=1.0, eps_end=0.1, eps_steps=1e6, discount=0.99, buffer_size=100000, batch_size=32, copy_times=1000):
        """
        DQNAlgo --- Implementation of simple deep Q-learning architecture
        setting parameters:
            filepath -- the data file used to create stocks trading environment
            netname -- 'fc' for fully connected network, 'conv1d' for 1D convolutional network
            algoname -- '' for naive dqn, 'duel' for dueling dqn
        """
        if netname == 'conv1d':
            env = StocksEnv.from_dir([filepath], state_1d=True)
        else:
            env = StocksEnv.from_dir([filepath])
        self.env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        if algoname == 'duel':
            netname = 'duel_' + netname
        self.agent = DQNAgent(netname, self.env.observation_space.shape, self.env.action_space.n, eps_start, discount)
        self.eps_start = eps_start
        self.eps_steps = eps_steps
        self.eps_end = eps_end
        self.batch_size = batch_size
        self.copy_times = 1000
        self.replay_buffer = deque(maxlen=buffer_size)
        self.iteration_n = 1
        self.rewards_record = []
        self.total_rewards_record = [0]

    def _get_batch(self):
        state_batch, action_batch, reward_batch, next_state_batch = [], [], [], []
        idxs = np.random.choice(len(self.replay_buffer), self.batch_size)
        for idx in idxs:
            s, a, r, next_s = self.replay_buffer[idx]
            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_state_batch.append(next_s)
        return np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch)

    def train_episode(self, max_iteration):
        state = self.env.reset()
        done = False
        while not done:
            action = self.agent.epsilon_greedy_action(state)
            next_s, reward, done, _ = self.env.step(action)
            self.replay_buffer.append((state, action, reward, next_s))
            if len(self.replay_buffer) >= self.batch_size:
                state_batch, action_batch, reward_batch, next_state_batch = self._get_batch()
                self.agent.update_eval_net(state_batch, reward_batch, action_batch, next_state_batch)
                if self.iteration_n % self.copy_times == 0:
                    self.agent.copy_to_target_net()
            
            self.rewards_record.append(reward)
            self.total_rewards_record.append(reward + self.total_rewards_record[-1])
            state = next_s
            self.agent.set_epsilon(self.eps_end - (self.eps_end - self.eps_start) / self.eps_steps * self.iteration_n)
            self.iteration_n += 1
            if self.iteration_n > max_iteration:
                break

    def train(self, max_iteration=1e7, save_pth='./model.pth'):
        self.iteration_n = 1
        self.rewards_record = []
        self.total_rewards_record = [0]
        while self.iteration_n <= max_iteration:
            self.train_episode(max_iteration)
        self.agent.save_model(save_pth)
        return self.rewards_record, self.total_rewards_record

    def valid(self, max_iteration=1e5, load_pth='./model.pth'):
        self.iteration_n = 1
        self.rewards_record = []
        self.total_rewards_record = [0]
        self.agent.load_model(load_pth)
        while self.iteration_n <= max_iteration:
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.greedy_action(state)
                next_s, reward, done, _ = self.env.step(action)
                self.rewards_record.append(reward)
                self.total_rewards_record.append(reward + self.total_rewards_record[-1])
                state = next_s
                self.iteration_n += 1
                if self.iteration_n > max_iteration:
                    break
                
        return self.rewards_record, self.total_rewards_record