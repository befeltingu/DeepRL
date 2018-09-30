# coding=utf-8
import gym
import numpy as np

from Player import DQNAgent,SimilarDQNAgent
from keras import backend as K




if __name__ == '__main__':

    EPISODES = 3000
    env = gym.make('LunarLander-v2')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    run_similarity_player = 1
    if run_similarity_player == 1:
        print("Running similarity player")

        agent = SimilarDQNAgent(state_size, action_size)

        agent.feature_layer = K.eval(agent.model.layers[0].weights[0])




    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    episode_scores = []
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_score = 0
        done = False
        while not done:
            #env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_score += reward
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, episode_score, agent.epsilon))

                episode_scores.append(episode_score)
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    np.save('/Users/befeltingu/DeepRL/Data/Gym/Lunar_reward.npy',np.array(episode_scores))