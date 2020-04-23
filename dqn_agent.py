import numpy as np
import torch
from model import *

class DQNAgent(object):
    def __init__(self, net_name, state_shape, action_dim, initial_epsilon, discount):
        """
        DQNAgent --- Implementation of simple dqn agent
        """
        assert isinstance(net_name, str) and net_name in ['fc', 'duel_fc', 'conv1d', 'duel_conv1d']
        assert isinstance(state_shape, int) or isinstance(state_shape, tuple)
        assert isinstance(action_dim, int)
        assert isinstance(initial_epsilon, float)
        assert isinstance(discount, float)
        self.eval_net = self._get_net(net_name, state_shape, action_dim)
        self.target_net = self._get_net(net_name, state_shape, action_dim)
        self.copy_to_target_net()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.0001)
        self.epsilon = initial_epsilon
        self.action_dim = action_dim
        self.discount = discount

    def _get_net(self, net_name, state_shape, action_dim):
        if net_name == 'fc':
            return FC(state_shape[0], action_dim)
        elif net_name == 'duel_fc':
            return Duel_FC(state_shape[0], action_dim)
        elif net_name == 'conv1d':
            return Conv_1D(state_shape, action_dim)
        elif net_name == 'duel_conv1d':
            return Duel_Conv_1D(state_shape, action_dim)

    def copy_to_target_net(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def update_eval_net(self, state_batch, reward_batch, action_batch, next_state_batch):
        batch_size = state_batch.shape[0]
        state_as_tensor = torch.from_numpy(state_batch)
        next_state_as_tensor = torch.from_numpy(next_state_batch)
        reward_as_tensor = torch.from_numpy(reward_batch).unsqueeze(1)
        one_hot_action = np.eye(self.action_dim)[action_batch]
        action_as_tensor = torch.from_numpy(one_hot_action)
        self.eval_net.train()
        self.optimizer.zero_grad()
        predict_v = self.eval_net(state_as_tensor)
        next_v, _ = torch.max(self.target_net(next_state_as_tensor), dim=1, keepdim=True)
        target_v = self.discount * next_v + reward_as_tensor
        label_v = (1.0 - action_as_tensor) * predict_v + action_as_tensor * target_v
        loss = torch.sum((label_v - predict_v) ** 2) / batch_size
        loss.backward()
        self.optimizer.step()

    def epsilon_greedy_action(self, state):
        if np.random.rand() > self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            self.eval_net.eval()
            state_as_tensor = torch.from_numpy(state).unsqueeze(dim=0)
            values = self.eval_net(state_as_tensor).detach().numpy()
            return np.argmax(values.reshape(-1))

    def greedy_action(self, state):
        self.eval_net.eval()
        state_as_tensor = torch.from_numpy(state).unsqueeze(dim=0)
        values = self.eval_net(state_as_tensor).detach().numpy()
        return np.argmax(values.reshape(-1))

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def save_model(self, filepath):
        torch.save(self.eval_net.state_dict(), filepath)

    def load_model(self, filepath):
        self.eval_net.load_state_dict(torch.load(filepath))