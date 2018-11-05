# coding=utf-8
# coding=utf-8
import numpy as np
import torch
import json
import pandas as pd

import torch.nn.functional as F

from Poker.AKQ.agents import PokerAgent,PokerAgent2

# Defining a Game simulation and getting rid of our Tree structures
class Game:

    def __init__(self):

        self.actions = ['bet', 'check', 'call', 'fold']
        self.anticipatory = 0.1
        self.GameState = None  # tensor that tracks current state
        self.current_player = None
        self.current_pot = 1.0  # starting amount = 1
        self.hands = [0, 1, 2]
        self.hand_string = ['A', 'K', 'Q']
        self.SB = None
        self.BB = None
        self.terminal_state = False

    def deal(self):

        return np.random.choice(self.hands, 2, replace=False)

    def display_state_prediction(self, state):

        current_policy = {
            "bet": 0,
            "check": 0
        }

        state = torch.from_numpy(state).float().unsqueeze(0)

        with torch.no_grad():

            action_values = self.SB.qnetwork_local(state)

        action_values = action_values.data.numpy()[0][:2]

        for i in range(2):
            current_policy[self.actions[i]] = action_values[i]

        print("Current local Q values")

        print(current_policy)

        with torch.no_grad():

            action_values = self.SB.qnetwork_target(state)

        action_values = action_values.data.numpy()[0][:2]

        for i in range(2):
            current_policy[self.actions[i]] = action_values[i]

        print("Current target Q values")

        print(current_policy)

        with torch.no_grad():

            action_values = self.SB.policynetwork(state)

        probs = F.softmax(action_values[0][:2].reshape((1, 2)))

        probs = probs.data.numpy()[0]

        for i in range(2):
            current_policy[self.actions[i]] = probs[i]

        print("Current Policy")

        print(current_policy)

    def display_experience_replay(self):
        '''
        Use for debugging to see whats going on inside the network
        :return:
        '''

        print("Displaying RL experience replay")

        experiences = self.SB.rl_replay_memory.memory

        experience_list = []

        for state, action, possible_actions, reward, next_state, dones in experiences:
            state = state[0]
            hand_state = state[:3]
            hand_index = np.where(hand_state == 1)[0][0]
            hand = self.hand_string[hand_index]

            sb_action = state[3:7]
            sb_action_index = np.where(sb_action == 1)[0][0]
            sb_action = self.actions[sb_action_index]

            experience_list.append([hand, sb_action, reward])

        rl_pands_df = pd.DataFrame(experience_list, columns=["Hand", "Action", "Reward"])

        print("Displaying SL experience replay")

        sl_experiences = self.SB.sl_replay_memory.memory

        sl_experience_list = []

        for state, action, possible_actions in sl_experiences:
            state = state[0]
            hand_state = state[:3]
            hand_index = np.where(hand_state == 1)[0][0]
            hand = self.hand_string[hand_index]

            sb_action = self.actions[action]

            sl_experience_list.append([hand, sb_action])

        sl_pandas_df = pd.DataFrame(sl_experience_list, columns=["Hand", "Action"])

        print("Done displaying experience")

    def get_action(self, current_state):

        if self.GameState.current_player.name == 'SB':

            possible_actions = self.get_possible_actions()

            return self.GameState.current_player.action(current_state,possible_actions),possible_actions

        elif self.GameState.current_player.name == 'BB':

            possible_actions = self.get_possible_actions()

            return self.GameState.current_player.action(current_state,possible_actions),possible_actions

    def get_next_state(self, a):

        '''
        The Game should be able to receive an action
        from one of the players and be able to update the current
        game state
        For the simple AKQ the only information the game needs is
        (player,action)
        In a more complex environment like LHE the game needs to be
        able to know more information
        (player, round, num_raises,action) + (board repr)

        '''

        # print("Get next state")
        # print("CUrrent state: ")
        # print(str(self.GameState.game_state))
        if self.GameState.current_player.name == 'SB':

            self.GameState.game_state[0][a] = 1

            if self.actions[a] == 'bet':
                self.GameState.current_pot += 1.0

                self.SB.stack_size -= 1.0

            self.GameState.current_player.name = 'BB'

        elif self.GameState.current_player.name == 'BB':

            self.GameState.game_state[1][a] = 1

            if self.actions[a] == 'call':
                self.GameState.current_pot += 1.0
                self.BB.stack_size -= 1.0

            self.GameState.current_player.name = 'SB'

            self.terminal_state = True

        return self.GameState.game_state

    def get_possible_actions(self):
        '''
        from the current game state what are the possible actions that
        the player can take
        :return:
        '''

        if self.GameState.current_player.name == 'SB':

            return [0,1]

        elif self.GameState.current_player.name == 'BB':

            sb_action = np.where(self.GameState.game_state[0] == 1)[0]

            if sb_action == 0: # sb bet so we can either call or fold

                return [2,3]

            elif sb_action == 1: # sb checked so we can bet or check

                return [0,1]

    def get_winning_player(self):

        if self.SB.hand < self.BB.hand:

            return self.SB
        else:
            return self.BB

    def init_game(self, SB, BB, GameState):

        self.GameState = GameState
        # SB = Player(name="SB", hands=self.hands, game_state=self.GameState)

        # BB = Player(name="BB", hands=self.hands, game_state=self.GameState)

        self.SB = SB

        self.BB = BB

    def reward(self):

        '''
            For the AKQ simple game this is... well simple but in general
            we will need to do an evaluation based off of how the episode
            was terminated.
        '''

        # get last 'node' in game state where action > 0

        i, j = np.where(self.GameState.game_state == 1)

        terminal_action_index = j[-1]

        terminal_row_index = i[-1]

        terminal_action = self.actions[terminal_action_index]

        winning_player = self.get_winning_player()

        reward = {
            'SB': 0.0,
            'BB': 0.0
        }

        # print("SB hand: {}".format(self.hand_string[self.SB.hand]))

        # print("BB hand: {}".format(self.hand_string[self.BB.hand]))

        rows , cols = np.where(self.GameState.game_state == 1)

        final_state , final_action = rows[-1], cols[-1]


        # print("SB action {}".format(sb_action))

        # handle case where there was a fold
        # the reward will go the current player because the previous player is the one that folded

        if final_action == 3:

            reward[self.GameState.current_player.name] = self.GameState.current_pot + self.SB.stack_size

            reward[self.GameState.current_player.name] = self.BB.stack_size

            return reward


        if winning_player.name == 'SB':

            reward['SB'] = self.GameState.current_pot + self.SB.stack_size
            reward['BB'] = self.BB.stack_size
        else:
            reward['BB'] = self.GameState.current_pot + self.BB.stack_size
            reward['SB'] = self.SB.stack_size

        # print(reward)

        return reward

    def run_sim(self, iters):

        for iter in range(iters):

            if (iter % 500) == 0:
                print("Iteration {}".format(str(iter)))

                #self.display_experience_replay()

                self.show_sb_policy()

            # print("Running iteration: {}".format(str(iter)))

            self.terminal_state = False

            self.SB.stack_size = 1.0
            self.BB.stack_size = 1.0

            hand1, hand2 = self.deal()

            self.SB.hand = hand1

            self.BB.hand = hand2

            game_state_init = np.zeros((2, 4))

            self.GameState.current_pot = 1.0

            self.GameState.game_state = game_state_init

            self.GameState.current_player = self.SB

            # set policy
            uniform = np.random.uniform(1)

            if self.anticipatory <= uniform:

                self.SB.current_policy = 'epsilon-greedy'

            else:

                self.SB.current_policy = 'policy'

            r = self.simulate()

            # update networks every ith step

        print("End of game")

    def simulate(self):

        current_state = self.GameState.game_state.copy()

        if self.terminal_state == True:

            return self.reward()

        action,possible_actions = self.get_action(current_state)

        next_state = self.get_next_state(action)

        done = self.terminal_state

        r = self.simulate()

        #self.GameState.current_player.step((current_state, action,possible_actions, r[self.GameState.current_player.name], next_state, done))

        self.SB.step((current_state, action,possible_actions, r['SB'], next_state, done))

        self.BB.step((current_state, action,possible_actions, r['BB'], next_state, done))

        return r

    def show_sb_policy(self):

        '''
        Test to make sure the sb is learning the game
        as expected
        :return:
        '''

        self.SB.policynetwork.eval()
        self.SB.qnetwork_local.eval()
        self.SB.qnetwork_target.eval()

        game_state_init = np.zeros((2, 4)).flatten()

        print("****** Displaying Policy ************")
        print("   ")
        print("SB Ace policy")
        A_hand = np.array([1, 0, 0])

        A_hand_state = np.concatenate((A_hand, game_state_init))

        self.display_state_prediction(A_hand_state)

        print("   ")
        print("SB King policy")
        print("   ")
        K_hand = [0, 1, 0]

        K_hand_state = np.concatenate((K_hand, game_state_init))

        self.display_state_prediction(K_hand_state)
        print("   ")
        print("SB Queen policy")
        print("   ")
        Q_hand = [0, 0, 1]

        Q_hand_state = np.concatenate((Q_hand, game_state_init))

        self.display_state_prediction(Q_hand_state)

        self.SB.policynetwork.train()
        self.SB.qnetwork_local.train()
        self.SB.qnetwork_target.train()


class GameState:
    '''
    Allow us to be able to pass state between classes
    '''

    def __init__(self, game_state, current_player, pot):
        self.game_state = game_state  # tensor that tracks current state
        self.current_player = current_player
        self.pot = pot  # starting amount = 1


def run_nfsp():

    STATE_SIZE = 11
    ACTION_SIZE = 4
    SEED = 1
    ITERS = 10000

    game_state_init = np.zeros((2, 4))

    State = GameState(game_state_init, current_player=None, pot=1.0)

    NFSPGame = Game()

    player1 = PokerAgent2(STATE_SIZE, ACTION_SIZE, SEED, State, NFSPGame.hands, 'SB', 1.0)

    player2 = PokerAgent2(STATE_SIZE, ACTION_SIZE, SEED, State, NFSPGame.hands, 'BB', 1.0)

    NFSPGame.init_game(player1, player2, State)

    NFSPGame.run_sim(ITERS)


if __name__ == '__main__':
    run_nfsp()