# coding=utf-8

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from collections import namedtuple, deque
from Poker.AKQ.networks import QNetwork,PolicyNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
RL_LR = 0.01  # learning rate
SL_LR = 0.005
LR = 5e-2
UPDATE_EVERY = 1  # how often to update the network
SOFT_UPDATE_EVERY = 8


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PokerAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed,game_state,hands,name,stack_size):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.current_policy = None
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.eps = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.9995

        # Poker specific attributes
        self.game_state = game_state
        self.hand = None  # index of hand 0,1 or 2
        self.board = None  # vector length 3
        self.hands = hands
        self.name = name
        self.stack_size = stack_size

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.q_optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=RL_LR)

        # Policy Network
        self.policynetwork = PolicyNetwork(state_size,action_size,seed).to(device)
        self.p_optimizer = optim.Adam(self.policynetwork.parameters(),lr=SL_LR)

        # Replay memory. Circular buffer
        self.rl_replay_memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Surpised learning. Reservoir
        self.sl_replay_memory = ReservoirBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, game_state):

        self.save_state(game_state)

        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1

        if (self.t_step % UPDATE_EVERY) == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.rl_replay_memory) > BATCH_SIZE:

                experiences = self.rl_replay_memory.sample()

                self.learn_q(experiences, GAMMA)

            # If enough samples are available in memory, get random subset and learn
            if len(self.sl_replay_memory) > BATCH_SIZE:

                experiences = self.sl_replay_memory.sample()

                self.learn_policy(experiences)

        self.eps = max(self.eps_end, self.eps_decay * self.eps)

    def action(self,state,possible_actions):


        if self.current_policy == 'epsilon-greedy':

            return self.act_greedy(state,possible_actions,self.eps)

        elif self.current_policy == 'policy':

            return self.act_policy(state,possible_actions,self.eps)

    def act_policy(self,state,possible_actions,eps=0.):

        state = self.convert_state(state)

        self.policynetwork.eval()

        with torch.no_grad():
            action_values = self.policynetwork(state)

        for i in range(4):
            if i not in possible_actions:
                action_values[0][i] = -1000

        self.policynetwork.train()

        return np.argmax(action_values.cpu().data.numpy())

    def act_greedy(self, state,possible_actions, eps=0.):

        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        state = self.convert_state(state)

        self.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        for i in range(4):
            if i not in possible_actions:
                action_values[0][i] = -1000

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(possible_actions)

    def convert_state(self,state):
        '''
        Take in game state and append players state
        :param state:
        :return:
        '''

        state = state.flatten()

        #state = state.reshape((state.shape[0]))

        hand = np.zeros(6)

        hand[self.hand] = 1

        board_card = np.where(self.board == 1)[0]

        if len(board_card) > 0:
            hand[board_card[0]] = 1

        state = np.concatenate((hand, state))

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        return state

    def learn_q(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.q_optimizer.zero_grad()

        loss.backward()
        self.q_optimizer.step()


        if (self.t_step % SOFT_UPDATE_EVERY) == 0:
            # ------------------- update target network ------------------- #

            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def learn_policy(self,experiences):

        states,actions = experiences

        predict_actions = self.policynetwork(states)

        predict_prob = F.softmax(predict_actions)

        loss = F.nll_loss(predict_prob,actions.reshape((predict_prob.shape[0])))

        self.p_optimizer.zero_grad()
        loss.backward()
        self.p_optimizer.step()

    def save_state(self,games_state):
        '''
        Take in state from the game and then add the players
        own information to the game state and save it in memory
        for latter training
        :param games_state:
        :return:
        '''
        state, action, reward, next_state, board, done = games_state

        state = self.convert_state(state)

        next_state = self.convert_state(next_state)

        # Save experience in replay memory
        self.rl_replay_memory.add(state, action, reward, next_state, done)

        if self.current_policy == 'epsilon-greedy':  # dont store if we are 'on policy'
            # Save experience in replay memory
            self.sl_replay_memory.add(state, action)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self,state, action,reward, next_state, done):

        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)

        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions,rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class ReservoirBuffer:

    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action"])
        self.seed = random.seed(seed)

    def add(self,state, action):
        """Add a new experience to memory."""
        e = self.experience(state, action)

        self.memory.append(e)

    def sample(self):

        """ Perform reservoir sampling """

        reservoir = []
        for i in range(self.batch_size):

            reservoir.append(self.memory[i])

        for i in range(self.batch_size,len(self.memory)):

            j= random.randrange(i + 1)

            if j < self.batch_size:

                reservoir[j] = self.memory[i]

        states = torch.from_numpy(np.vstack([e.state for e in reservoir if e is not None])).float().to(device)

        actions = torch.from_numpy(np.vstack([e.action for e in reservoir if e is not None])).long().to(device)

        return (states, actions)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

