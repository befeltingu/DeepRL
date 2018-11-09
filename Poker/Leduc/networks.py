
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size,seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.action_size = action_size
        self.model = None
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.init()


    def init(self):

        model = nn.Sequential(OrderedDict([
            ('ds1',nn.Linear(self.state_size,64)),
            ('act1', nn.ReLU()),
            ('ds2', nn.Linear(64,self.action_size))

        ]))

        self.model = model


    def forward(self, state):

        return self.model(state)

class PolicyNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size,seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(PolicyNetwork, self).__init__()
        self.action_size = action_size
        self.model = None
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.init()


    def init(self):

        model = nn.Sequential(OrderedDict([
            ('ds1', nn.Linear(self.state_size, 64)),
            ('act1', nn.ReLU()),
            ('ds2', nn.Linear(64, self.action_size))

        ]))

        self.model = model


    def forward(self, state):

        return self.model(state)

