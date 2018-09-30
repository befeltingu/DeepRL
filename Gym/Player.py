# coding=utf-8

import gym
import numpy as np
import pandas as pd
import random


from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity




def get_similar_vectors(state,state_feature_vector ,layer_weights ,num_samples ,top_n):
    '''
        Takes in a state and returns the top x
        most similar states
    '''

    state = list(state[0])
    sampled_feature_vectors, sample_states = sample_state_space(state,num_samples,layer_weights)

    cosine_scores = []
    for i ,j in enumerate(sampled_feature_vectors):

        cosine_score = cosine_similarity(state_feature_vector ,j)

        cosine_scores.append([i ,cosine_score])

    cosine_score_df = pd.DataFrame(cosine_scores ,columns=["index" ,"score"])

    cosine_score_df = cosine_score_df.sort_values(by=["score"] ,ascending=False).iloc[:top_n]

    top_indexes = cosine_score_df["index"].tolist()

    top_samples = [sample_states[top_index] for top_index in top_indexes]

    return top_samples


def sample_state_space(state,num_samples, layer_weights):
    '''
        Want to be able to randomly sample from the space
        this will be game dependent. In the case of blackJack
        Just sample a dealer hand and then a random sum for the
        player between 4 and 21 and then sample
    '''
    sample_list = []
    while len(sample_list) < num_samples:

        dealer_hand = np.random.randint(1, 11)
        player_hand = np.random.randint(4, 21)
        ace_or_not = np.random.binomial(1, (1.0 / 13.0))

        sampled_state = [dealer_hand, player_hand, ace_or_not]

        if (sampled_state == state) or (sampled_state in sample_list): # dont sample the same state as your interested in
            continue

        sample_list.append(sampled_state)

    feature_vectors = np.matmul(np.array(sample_list), layer_weights)

    return feature_vectors, sample_list



class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



class SimilarDQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.feature_layer = None
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:

            target = reward

            if not done:

                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)

            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

            # using the same reward get a bunch of similar states and fit the model with the expectation that
            # similar states would lead to similar outcomes especailly in the case when the episode is finished.

            state_feature_vector = np.matmul(state, self.feature_layer)

            similar_states = get_similar_vectors(state,state_feature_vector,self.feature_layer,200,10)

            if done:
                for similar_state in similar_states:

                    similar_state = np.reshape(similar_state,state.shape)

                    target_f = self.model.predict(similar_state)

                    target_f[0][action] = target

                    self.model.fit(similar_state, target_f, epochs=1, verbose=0)



        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
