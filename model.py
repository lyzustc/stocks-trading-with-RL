import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class Duel_FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Duel_FC, self).__init__()
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.adv_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        value = self.value_net(x)
        adv = self.adv_net(x)
        out = value + (adv - adv.mean(dim=1, keepdims=True))
        return out