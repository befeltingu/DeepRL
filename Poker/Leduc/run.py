# coding=utf-8
import numpy as np
import torch
import json
import pandas as pd

import torch.nn.functional as F

from Poker.Leduc.agents import PokerAgent, LeducAgent, RandomJack


# Defining a Game simulation and getting rid of our Tree structures
class Game:

    def __init__(self):

        self.actions = ['raise', 'check', 'call', 'fold']
        self.anticipatory = 0.2
        self.GameState = None  # tensor that tracks current state
        self.current_player = None
        self.current_pot = 1.0  # starting amount = 1
        self.hands = [0, 1, 2, 3, 4, 5]
        self.hand_string = ['As', 'Ad', 'Ks', 'Kd', 'Qs', 'Qd']
        self.SB = None
        self.BB = None
        self.players = ['_', self.SB, self.BB]

    def check_terminal(self, state):
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

    def display_state_prediction(self, state):

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

            experience_list.append([hand, sb_action, reward])

        rl_pands_df = pd.DataFrame(experience_list, columns=["Hand", "Action", "Reward"])

        print("Displaying SL experience replay")

        sl_experiences = self.SB.sl_replay_memory.memory

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

    def evaluate_policy(self):
        '''
        We will evaluate the policy by looking at exploitability
        This is defined as the expected average payoff that a best response profile achieves against it.
        :return:
        '''

        # States (player,round,num raises,action)
        # (2,2,3,2)
        # For sb
        # for i in range(2):
        #    for j in range(3):
        #        for k in range(2):
        #            state = torch.zeros((2,2,3,2))
        #            #state[]
        pass

    def evaluate_policy_simple(self, iters, hero,opponent):

        """
        Just going to use predefined s
        :return:
        """


        reward_tot = 0

        for iter in range(iters):

            self.hands = [0, 1, 2, 3, 4, 5]

            self.terminal_state = False

            hero.stack_size = 5.0
            opponent.stack_size = 5.0

            hand1, hand2 = self.deal()

            hero.hand = hand1

            opponent.hand = hand2

            self.hands.remove(hand1)
            self.hands.remove(hand2)

            if hero.name == 'SB':
                self.GameState.reset(hero)
            else:
                self.GameState.reset(opponent)

            hero.current_policy = 'policy'

            r = self.evaluate_simulate()

            reward_tot += r[hero]

        print("Evaluation for hero = {} ".format(str(hero)))
        print("Average reward {}".format(str(reward_tot / float(iters))))

        print("End of game")

    def evaluate_reward(self, previous_action):
        """
        :return:
        """
        current_player = self.GameState.current_player
        current_player_name = current_player.name
        current_player_stack = current_player.stack_size

        if current_player_name == 'SB':

            opponent_player = self.BB
            opponent_player_name = opponent_player.name
            opponent_player_stack = opponent_player.stack_size

        elif current_player_name == 'BB':

            opponent_player = self.SB
            opponent_player_name = opponent_player.name
            opponent_player_stack = opponent_player.stack_size

        current_pot = self.GameState.current_pot

        r = {
            current_player_name: 0,
            opponent_player_name: 0
        }

        winning_player, losing_player = self.get_winning_player()

        if self.actions[previous_action] == 'fold':

            # current player gets the pot
            r[current_player_name] = current_player_stack

            r[opponent_player_name] = current_pot + opponent_player_stack

        elif self.actions[previous_action] == 'call' or self.actions[previous_action] == 'check':

            r[winning_player.name] = current_pot + winning_player.stack_size
            r[losing_player.name] = losing_player.stack_size

        return r

    def evaluate_simulate(self, action=None):
        """
        :param action:
        :return:
        """

        current_state = self.GameState.game_state.copy()

        board = self.GameState.board.copy()

        current_player = self.GameState.current_player.name

        if self.GameState.terminal_state:

            return self.evaluate_reward(action)

        action = self.get_action(current_state)

        next_state = self.get_next_state(action).copy()

        done = self.GameState.terminal_state

        r = self.evaluate_simulate(action=action)

        return r

    def get_action(self, current_state):
        """

        :param current_state:
        :return:
        """

        if self.GameState.current_player.name == 'SB':

            possible_actions = self.get_possible_actions('SB')

        elif self.GameState.current_player.name == 'BB':

            possible_actions = self.get_possible_actions('BB')

        return self.GameState.current_player.action(current_state, possible_actions)

    def get_possible_actions(self, player):
        """
        actions ['raise', 'check', 'call', 'fold']
        :param player:
        :return:
        """

        """

                if player == 'SB':

                    if self.GameState.current_round == 0:

                        if self.GameState.num_raises == 1:
                            return [0,2,3]

                        elif self.GameState.num_raises == 0:
                            return [0,1]

                        elif self.GameState.num_raises >= 2:
                            return [2,3]

                    elif self.GameState.current_round == 1:

                        if self.GameState.num_raises == 1:
                            return [0, 2, 3]

                        elif self.GameState.num_raises == 0:
                            return [0, 1]

                        elif self.GameState.num_raises >= 2:
                            return [2, 3]

                """

        if self.GameState.num_raises == 1:
            return [0, 2, 3]

        elif self.GameState.num_raises == 0:
            return [0, 1]

        elif self.GameState.num_raises >= 2:
            return [2, 3]
        return

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

        current_round = self.GameState.current_round
        #print("Current round {}".format(str(current_round)))
        #print("Player {} takes action {} ".format(self.GameState.current_player.name, self.actions[a]))

        a_repr = a
        if a == 2:
            a_repr = 1  # check and call are the same in the state representation

        if self.GameState.current_player.name == 'SB':

            if self.actions[a] == 'raise':

                self.GameState.num_raises += 1.0

                self.GameState.current_pot += 1.0

                self.SB.stack_size -= 1.0

            elif self.actions[a] == 'call':

                self.GameState.current_pot += 1.0

                self.SB.stack_size -= 1.0

                self.update_current_round()

            elif self.actions[a] == 'check':

                # if current round 0 then action goes to BB
                if self.GameState.current_round == 0:

                    self.GameState.current_player = self.BB

                else:
                    self.update_current_round()

            elif self.actions[a] == 'fold':

                self.GameState.terminal_state = True

            if self.actions[a] != 'fold':
                # update state now that all actions have been porcessed
                self.GameState.game_state[0][current_round][int(self.GameState.num_raises)][a_repr] = 1
            # switch the current player
            self.GameState.current_player = self.BB

        elif self.GameState.current_player.name == 'BB':

            if self.actions[a] == 'raise':

                self.GameState.num_raises += 1.0

                self.GameState.current_pot += 1.0

                self.BB.stack_size -= 1.0

                # switch the current player
                self.GameState.current_player = self.SB

            elif self.actions[a] == 'call':

                self.GameState.current_pot += 1.0

                self.BB.stack_size -= 1.0

                self.update_current_round()

            elif self.actions[a] == 'check':
                # If BB checks Do nothing. Below we will update the current player
                # switch the current player

                # if current round 0 then action goes to BB
                if self.GameState.current_round == 0:

                    self.update_current_round()

                else:
                    self.GameState.current_player = self.SB

            elif self.actions[a] == 'fold':

                self.GameState.terminal_state = True

            if self.actions[a] != 'fold':
                # update state now that all actions have been porcessed
                self.GameState.game_state[1][current_round][int(self.GameState.num_raises)][a_repr] = 1

        if self.GameState.current_round == 2:
            self.GameState.terminal_state = True

        return self.GameState.game_state

    def get_winning_player(self):
        """

        :return: tuple (winning, losing)
        """

        board = np.where(self.GameState.board == 1)[0]

        if len(board) > 0:

            board_str = self.hand_string[board[0]]
            sb_hand_str = self.hand_string[self.SB.hand]
            bb_hand_str = self.hand_string[self.BB.hand]

            if sb_hand_str[0] == board_str[0]:

                return self.SB, self.BB

            elif bb_hand_str[0] == board_str[0]:

                return self.SB, self.BB

        # neither players hand matches the board
        if self.SB.hand < self.BB.hand:

            return self.SB, self.BB
        else:
            return self.BB, self.SB

    def init_game(self, SB, BB, GameState):

        self.GameState = GameState
        # SB = Player(name="SB", hands=self.hands, game_state=self.GameState)

        # BB = Player(name="BB", hands=self.hands, game_state=self.GameState)

        self.SB = SB

        self.BB = BB

    def reward(self, previous_action=None):
        """
            get last action that was taken

            Fold:
                Then reward current player since the player that folded is no longer current player

            Call:
                Doesnt matter who called. Just evaluate hand v hand

            Check:
                Again just evaluate hand v hand
        """

        # get last 'node' in game state where action > 0

        current_player = self.GameState.current_player
        current_player_name = current_player.name
        current_player_stack = current_player.stack_size

        opponent_player = self.players[-1 * self.players.index(current_player)]
        opponent_player_name = opponent_player.name
        opponent_player_stack = opponent_player.stack_size

        current_pot = self.GameState.current_pot

        r = {
            current_player_name: 0,
            opponent_player_name: 0

        }

        winning_player, losing_player = self.get_winning_player()

        if self.actions[previous_action] == 'fold':

            # current player gets the pot
            r[current_player_name] = current_player_stack

            r[opponent_player_name] = current_pot + opponent_player_stack

        elif self.actions[previous_action] == 'call' or self.actions[previous_action] == 'check':

            r[winning_player.name] = current_pot + winning_player.stack_size
            r[losing_player.name] = losing_player.stack_size

        return r

    def run_sim(self, iters):

        for iter in range(iters):

            #print("Episode {} ".format(str(iter)))
            #print("*******************************")

            if (iter % 1000) == 0:
                print("Evaluating model")
                print("Iteration {}".format(str(iter)))

                hero = self.SB
                opponent = RandomJack(self.GameState, self.hands, 'BB', 5.0)

                self.evaluate_policy_simple(1000, hero,opponent)

                hero = self.BB
                opponent = RandomJack(self.GameState, self.hands, 'SB', 5.0)

                self.evaluate_policy_simple(1000, hero,opponent)

            self.hands = [0, 1, 2, 3, 4, 5]

            self.terminal_state = False

            self.SB.stack_size = 5.0
            self.BB.stack_size = 5.0

            hand1, hand2 = self.deal()

            self.SB.hand = hand1

            self.BB.hand = hand2

            self.hands.remove(hand1)
            self.hands.remove(hand2)

            #print("New Deal")
            #print("SB Hand = {}".format(self.hand_string[hand1]))
            #print("BB Hand = {}".format(self.hand_string[hand2]))

            self.GameState.reset(self.SB)

            # set policy
            uniform = np.random.uniform(1)

            if self.anticipatory <= uniform:

                self.SB.current_policy = 'epsilon-greedy'
                self.BB.current_policy = 'epsilon-greedy'

            else:

                self.SB.current_policy = 'policy'
                self.BB.current_policy = 'policy'

            r = self.simulate()

            #print("Reward for round")
            #print(str(r))

            # update networks every ith step

        print("End of game")

    def simulate(self, action=None):

        current_state = self.GameState.game_state.copy()

        board = self.GameState.board.copy()

        current_player = self.GameState.current_player.name

        if self.GameState.terminal_state:
            return self.reward(action)

        action = self.get_action(current_state)

        next_state = self.get_next_state(action).copy()

        done = self.GameState.terminal_state

        r = self.simulate(action=action)

        if current_player == 'SB':

            self.SB.step((current_state, action, r['SB'], next_state, board, done))

        elif current_player == 'BB':

            self.BB.step((current_state, action, r['BB'], next_state, board, done))

        return r

    def show_sb_policy(self):

        """
        Test to make sure the sb is learning the game
        as expected
        :return:
        """

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

    def update_current_round(self):
        """
        helper function to update the current round
        and reset number of raises
        :return:
        """

        self.GameState.num_raises = 0

        if self.GameState.current_round == 1:

            self.GameState.terminal_state = True

        else:

            flop_card = np.random.choice(self.hands)

            print("Flop {} ".format(str(self.hand_string[flop_card])))

            self.GameState.board[np.random.randint(6)] = 1  # draw random card to represent the board

            self.GameState.current_round += 1


class GameState:
    """
    Allow us to be able to pass state between classes
    """

    def __init__(self, game_state, current_player, pot, current_policy='epsilon-greedy'):
        self.board = np.zeros(6)
        self.current_player = current_player
        self.current_policy = current_policy
        self.current_round = 0
        self.game_state = game_state  # tensor that tracks current state
        self.num_raises = 0
        self.current_pot = pot  # starting amount = 1
        self.terminal_state = False

    def reset(self, player):
        self.board = np.zeros(6)
        self.current_player = player
        self.current_round = 0
        self.game_state = np.zeros((2, 2, 3, 2))
        self.num_raises = 0
        self.terminal_state = False
        self.current_pot = 1


def run_nfsp():
    STATE_SIZE = 30
    ACTION_SIZE = 4
    SEED = 1
    ITERS = 50000

    # Game state { player, round, num raises, action}
    game_state_init = np.zeros((2, 2, 3, 2))

    State = GameState(game_state_init, current_player=None, pot=1.0)

    NFSPGame = Game()

    player1 = PokerAgent(STATE_SIZE, ACTION_SIZE, SEED, State, NFSPGame.hands, 'SB', 5.0)

    player2 = PokerAgent(STATE_SIZE, ACTION_SIZE, SEED, State, NFSPGame.hands, 'BB', 5.0)

    NFSPGame.init_game(player1, player2, State)

    NFSPGame.run_sim(ITERS)


if __name__ == '__main__':
    run_nfsp()
