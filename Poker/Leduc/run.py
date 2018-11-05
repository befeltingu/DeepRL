
# coding=utf-8
import numpy as np
import torch
import json
import pandas as pd

import torch.nn.functional as F

from Poker.Leduc.agents import PokerAgent


# Defining a Game simulation and getting rid of our Tree structures
class Game:

    def __init__(self):

        self.actions = ['raise', 'check', 'call', 'fold']
        self.anticipatory = 0.2
        self.GameState = None  # tensor that tracks current state
        self.current_player = None
        self.current_pot = 1.0  # starting amount = 1
        self.hands = [0, 1, 2]
        self.hand_string = ['A', 'K', 'Q']
        self.SB = None
        self.BB = None
        self.terminal_state = False

    def check_terminal(self,state):
        '''
        Function to check whether the state is terminal or not
        Conditions:
            A: The state is terminal if the last round with an action
            taken is > # of rounds in game.

            B: If current round is max round and number raises == max raises

        :Input: Tensor (2,2,3,2)
        :return: Boolean
        '''

        # Condition A
        pass

    def deal(self):

        return np.random.choice(self.hands, 2, replace=False)

    def display_state_prediction(self,state):

        current_policy = {
            "raise": 0,
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

        for state, action, reward, next_state, dones in experiences:
            state = state[0]
            hand_state = state[:3]
            hand_index = np.where(hand_state == 1)[0][0]
            hand = self.hand_string[hand_index]

            sb_action = state[3:8]
            sb_action_index = np.where(sb_action == 1)[0][0]
            sb_action = self.actions[sb_action_index]

            experience_list.append([hand,sb_action,reward])

        rl_pands_df = pd.DataFrame(experience_list,columns=["Hand","Action","Reward"])

        print("Displaying SL experience replay")

        sl_experiences= self.SB.sl_replay_memory.memory

        sl_experience_list = []

        for state, action in sl_experiences:

            state = state[0]
            hand_state = state[:3]
            hand_index = np.where(hand_state == 1)[0][0]
            hand = self.hand_string[hand_index]

            sb_action = self.actions[action]

            sl_experience_list.append([hand, sb_action])

        sl_pandas_df = pd.DataFrame(sl_experience_list, columns=["Hand", "Action"])

        print("Done displaying experience")

    def get_action(self,current_state):

        if self.GameState.current_player.name == 'SB':

            return self.GameState.current_player.action(current_state)

        elif self.GameState.current_player.name == 'BB':

            sb_action = np.where(current_state[0] == 1.0)[0][0]

            if sb_action == 0:

                return 2  # force call

            elif sb_action == 1:
                return 1  # force check

            else:
                print("sb took illegal action")

    def get_next_state(self, a):
        """
        The Game should be able to receive an action
        from one of the players and be able to update the current
        game state.
        The Game state has a number of things that need to be tracked here
            -- Number raises
            -- Round number
            -- if its a leaf / terminal state
            -- Pot
            -- Player stacks

        """

        current_round = self.GameState.current_round.copy()

        if self.GameState.current_player.name == 'SB':

            if self.actions[a] == 'raise':

                self.GameState.number_raises += 1.0

                self.GameState.current_pot += 1.0

                self.SB.stack_size -= 1.0

            elif self.actions[a] == 'call':

                self.GameState.current_pot += 1.0

                self.SB.stack_size -= 1.0

                self.update_current_round()

            elif self.actions[a] == 'check':
                # if the sb checks that will always terminate a round
                self.update_current_round()

            elif self.actions[a] == 'fold':

                self.GameState.terminal_state = True

            # update state now that all actions have been porcessed
            self.GameState.game_state[0][current_round][self.GameState.num_raises][a] = 1
            # switch the current player
            self.GameState.current_player.name = 'BB'

        elif self.GameState.current_player.name == 'BB':

            if self.actions[a] == 'raise':

                self.GameState.number_raises += 1.0

                self.GameState.current_pot += 1.0

                self.BB.stack_size -= 1.0

            elif self.actions[a] == 'call':

                self.GameState.current_pot += 1.0

                self.BB.stack_size -= 1.0

                self.update_current_round()

            elif self.actions[a] == 'check':
                # If BB checks Do nothing. Below we will update the current player
                pass

            elif self.actions[a] == 'fold':

                self.update_current_round()

            # update state now that all actions have been porcessed
            self.GameState.game_state[1][current_round][self.GameState.num_raises][a] = 1
            # switch the current player
            self.GameState.current_player.name = 'SB'

        return self.GameState.game_state

    def get_winning_player(self):

        if self.SB.hand < self.BB.hand:

            return self.SB
        else:
            return self.BB

    def init_game(self,SB,BB,GameState):

        self.GameState = GameState
        #SB = Player(name="SB", hands=self.hands, game_state=self.GameState)

        #BB = Player(name="BB", hands=self.hands, game_state=self.GameState)

        self.SB = SB

        self.BB = BB

    def reward(self):

        '''
            get last action that was taken

            Fold:
                Then reward current player since the player that folded is no longer current player

            Call:
                Doesnt matter who called. Just evaluate hand v hand

            Check:
                Again just evaluate hand v hand
        '''

        # get last 'node' in game state where action > 0

        if self.GameState.current_player.name == 'SB': # SB is current that means BB was last to act

            i, j, k = np.where(self.GameState.game_state[1] == 1)

        elif self.GameState.current_player.name == 'BB': # BB is current that means SB was last to act

            i, j, k = np.where(self.GameState.game_state[0] == 1)

        terminal_action_index = j[-1]

        terminal_row_index = i[-1]

        terminal_action = self.actions[terminal_action_index]

        winning_player = self.get_winning_player()

        reward = {
            'SB': 0.0,
            'BB': 0.0
        }

        #print("SB hand: {}".format(self.hand_string[self.SB.hand]))

        #print("BB hand: {}".format(self.hand_string[self.BB.hand]))

        sb_action = np.where(self.GameState.game_state[0] == 1.0)[0][0]

        sb_action = self.actions[sb_action]

        #print("SB action {}".format(sb_action))


        if winning_player.name == 'SB':

            reward['SB'] = self.GameState.current_pot + self.SB.stack_size
            reward['BB'] = self.BB.stack_size
        else:
            reward['BB'] = self.GameState.current_pot + self.BB.stack_size
            reward['SB'] = self.SB.stack_size

        #print(reward)

        return reward

    def run_sim(self,iters):


        for iter in range(iters):

            if(iter % 1000) == 0:

                print("Iteration {}".format(str(iter)))

                #self.display_experience_replay()

                self.show_sb_policy()

            #print("Running iteration: {}".format(str(iter)))

            self.terminal_state = False

            self.SB.stack_size = 5.0
            self.BB.stack_size = 5.0

            hand1, hand2 = self.deal()

            self.SB.hand = hand1

            self.BB.hand = hand2

            self.GameState.reset(self.SB)

            # set policy
            uniform = np.random.uniform(1)

            if self.anticipatory <= uniform:

                self.GameState.current_policy = 'epsilon-greedy'

            else:

                self.GameState.current_policy = 'policy'

            r = self.simulate()

            # update networks every ith step

        print("End of game")

    def simulate(self):

        current_state = self.GameState.game_state.copy()

        board = self.GameState.board.copy()

        current_player = self.GameState.current_player.name

        if self.terminal_state:

            return self.reward()

        action = self.get_action(current_state)

        next_state = self.get_next_state(action).copy()

        done = self.terminal_state

        r = self.simulate()

        if current_player == 'SB':

            self.SB.step((current_state,action,r['SB'],next_state,board,done))

        elif current_player == 'BB':

            self.BB.step((current_state,action,r['BB'],next_state,board,done))

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
        A_hand = np.array([1,0,0])

        A_hand_state = np.concatenate((A_hand,game_state_init))

        self.display_state_prediction(A_hand_state)

        print("   ")
        print("SB King policy")
        print("   ")
        K_hand = [0,1,0]

        K_hand_state = np.concatenate((K_hand,game_state_init))

        self.display_state_prediction(K_hand_state)
        print("   ")
        print("SB Queen policy")
        print("   ")
        Q_hand = [0,0,1]

        Q_hand_state = np.concatenate((Q_hand,game_state_init))

        self.display_state_prediction(Q_hand_state)

        self.SB.policynetwork.train()
        self.SB.qnetwork_local.train()
        self.SB.qnetwork_target.train()

    def update_current_round(self):
        """
        helper function to update the current round
        and reset number of raises
        :return:
        """
        self.GameState.current_round += 1
        self.GameState.board[np.random.randint(3)] = 1 # draw random card to represent the board
        self.GameState.number_raises = 0

        if self.GameState.current_round > 1:

            self.GameState.terminal_state = True


class GameState:
    '''
    Allow us to be able to pass state between classes
    '''
    def __init__(self,game_state,current_player,pot,current_policy='epsilon-greedy'):

        self.board = np.zeros(3)
        self.current_player = current_player
        self.current_policy = current_policy
        self.current_round = 0
        self.game_state = game_state  # tensor that tracks current state
        self.num_raises = 0
        self.pot = pot  # starting amount = 1
        self.terminal_state = False

    def reset(self,player):

        self.board = np.zeros(3)
        self.current_player = player
        self.current_round = 0
        self.game_state = np.zeros((2, 2, 3, 2))
        self.num_raises = 0
        self.terminal_state = False
        self.pot = 1


def run_nfsp():

    STATE_SIZE = 30
    ACTION_SIZE = 4
    SEED = 1
    ITERS = 50000

    # Game state { player, round, num raises, action}
    game_state_init = np.zeros((2,2,3,2))

    State = GameState(game_state_init, current_player=None, pot=1.0)

    NFSPGame = Game()

    player1 = PokerAgent(STATE_SIZE, ACTION_SIZE, SEED, State, NFSPGame.hands, 'SB', 5.0)

    player2 = PokerAgent(STATE_SIZE, ACTION_SIZE, SEED, State, NFSPGame.hands, 'BB', 5.0)

    NFSPGame.init_game(player1,player2,State)

    NFSPGame.run_sim(ITERS)



if __name__ == '__main__':

    run_nfsp()