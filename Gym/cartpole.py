# coding=utf-8
import gym
import numpy as np

from Player import DQNAgent



if __name__ == '__main__':

    EPISODES = 100
    env = gym.make('CartPole-v1')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)



    # done training lets see how it does

    env = gym.wrappers.Monitor(env, '/Users/befeltingu/DeepRL/Data/Gym/cartpole-experiment-1', force=True, video_callable=lambda episode_id: True)

    num_failures = 0
    avg_time_to_death = 0
    for tries in range(5):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for i in range(500):

            env.render()
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            if done == True:
                num_failures += 1
                avg_time_to_death += i
                print("You died in {} tried".format(i))
                break
            state = np.reshape(state, [1, state_size])


    print("Avg dead time: " + str(avg_time_to_death / float(50)))
    print("Total fails: " + str(num_failures))
