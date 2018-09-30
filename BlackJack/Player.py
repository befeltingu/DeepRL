import random
import gym
import numpy as np
import pandas as pd

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K



# Lets build out our Deep RL player we are going to use the DQN model as described in Deepminds
# paper https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

class DeepPlayer:
    # simple DQN agent
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.actions = {0:'HIT' ,1:'STAY'}
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.65  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense( 4 *self.state_size, input_dim=self.state_size, activation='relu',use_bias=False))
        model.add(Dense( 4 *self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse' ,optimizer=Adam(lr=self.learning_rate))
        return model

    def create_policy_df(self):


        policy_array = []
        # loop over dealer up card
        for i in range(1,11):
            # loop over possible player totals
            for j in range(4,22):
                # whether player is holding playable ace
                for k in range(2):

                    state = np.array([[i,j,k]])

                    action = self.make_max_play(state)

                    policy_array.append([i,j,k,action])


        policy_df = pd.DataFrame(policy_array,columns=['Dealer','Player','Ace','Policy'])

        policy_df.to_pickle('/Users/befeltingu/DeepRL/Data/DataFrames/blackjack_policy0')

    def make_play(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def make_max_play(self,state):

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def get_action_number(self ,action_string):

        for key,value in self.actions.iteritems():

            if action_string == value:
                return key

        print("Error in get_action_number incorrect action string {}".format(action_string))

    def get_player_state(self ,cards):

        usable_ace = 0
        sum_cards = np.array(cards).sum()

        if 1 in cards:
            sum_cards_use_ace = sum_cards + 10
            if sum_cards_use_ace <= 21:
                usable_ace = 1
                sum_cards = sum_cards_use_ace

        return sum_cards, usable_ace

    def get_player_score(self,cards):

        sum_cards = np.array(cards).sum()

        if 1 in cards:
            sum_cards_use_ace = sum_cards + 10
            if sum_cards_use_ace <= 21:
                sum_cards = sum_cards_use_ace

        return sum_cards

    def load(self, name):
        self.model.load_weights(name)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            try:
                target_f[0][action] = target
            except Exception,e:
                print(e)
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)