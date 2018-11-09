# coding=utf-8

# coding=utf-8
import numpy as np
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn.functional as F

from Poker.Limit.agents import PokerAgent
from treys import Deck
from treys import Card
from treys import Evaluator


# Defining a Game simulation and getting rid of our Tree structures
class Game:

    def __init__(self):

        self.actions = ['raise', 'check', 'call', 'fold']
        self.anticipatory = 0.1
        self.GameState = None  # tensor that tracks current state
        self.current_player = None
        self.deck = Deck()
        self.deck_lookup = None
        self.evaluator = Evaluator()
        self.S = 1000  # Starting stack
        self.SB = None
        self.BB = None
        self.players = ['_', self.SB, self.BB]

    def deal(self):
        """
        :return:
        """
        self.SB.hand_bit = self.deck.draw(2)
        self.SB.hand = torch.zeros(52)
        self.SB.hand[self.deck_lookup[self.SB.hand_bit[0]]] = 1
        self.SB.hand[self.deck_lookup[self.SB.hand_bit[1]]] = 1

        self.BB.hand_bit = self.deck.draw(2)
        self.BB.hand = torch.zeros(52)
        self.BB.hand[self.deck_lookup[self.BB.hand_bit[0]]] = 1
        self.BB.hand[self.deck_lookup[self.BB.hand_bit[1]]] = 1

    def deal_next_round(self):
        """
        helper function to update the current round
        and reset number of raises
        :return:
        """
        if self.GameState.current_round == 0: # deal flop
            print("Dealing Flop")
            flop = self.deck.draw(3)
            print(Card.print_pretty_cards(flop))
            self.GameState.bit_flop = flop
            self.GameState.flop[self.deck_lookup[flop[0]]] = 1
            self.GameState.flop[self.deck_lookup[flop[1]]] = 1
            self.GameState.flop[self.deck_lookup[flop[2]]] = 1

        if self.GameState.current_round == 1:
            print("Dealing Turn")
            self.GameState.bb_size = int(self.GameState.bb_size*2)
            turn = self.deck.draw(1)
            print(Card.print_pretty_cards([turn]))
            self.GameState.bit_turn = [turn]
            self.GameState.turn[self.deck_lookup[turn]] = 1

        if self.GameState.current_round == 2:
            print("Dealing River")
            river = self.deck.draw(1)
            print(Card.print_pretty_cards([river]))
            self.GameState.bit_river = [river]
            self.GameState.flop[self.deck_lookup[river]] = 1

        # num raises starts over
        self.GameState.num_raises = 0

    def display_state_prediction(self, state):
        """

        :param state:
        :return:
        """

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

    def get_bit_board(self):

        if not self.GameState.bit_flop:
            return None

        if not self.GameState.bit_turn:

            return self.GameState.bit_flop

        if not self.GameState.bit_river:
            return self.GameState.bit_flop + self.GameState.bit_turn

        return self.GameState.bit_flop + self.GameState.bit_turn + self.GameState.bit_river

    def get_possible_actions(self, player):
        """
            Need:
                GameState:
                    -- previous_action
                    -- num raises
                    -- current round
        """
        if self.GameState.num_raises == 0:

            if self.GameState.current_round == 0: # SB starting
                # ['raise',call,'fold']
                if self.GameState.current_player.name == 'SB':
                    return [0,2,3]
                else:
                    return [0,1]
            else:
                # ['raise','check']
                return [0,1]

        if self.GameState.num_raises == 4:
            # ['call','fold']
            return [2,3]

        if self.GameState.num_raises > 0 and self.GameState.num_raises < 4:
            # ['raise','call','fold']
            return [0,2,3]
        return

    def get_next_state(self, action):
        """
        Need:
            State = {player x round x number raises x action }
            The game needs to know when an episode should terminate
            The game needs to know when a round has completed

            We will be ignoring for now conditions in which the player is
            all in previous to the river. This wont happen in our limit holdem environment
            anyway
        """

        if action == 'fold':
            # no need to update any states
            # just set the game state to terminal
            self.GameState.terminal_state = True
            return

        if self.GameState.current_round == 3 and action == 'call':
            # there is no next state but update the Game state
            self.GameState.terminal_state = True
            self.GameState.current_player -= self.GameState.bb_size
            self.GameState.current_pot += self.GameState.bb_size
            return

        if self.GameState.current_round == 0:
            if action == 'check' and self.GameState.previous_action == 'call':
                self.deal_next_round()
                self.update_game_state(action,next_round=True)
                return

            if action == 'call' and self.GameState.previous_action =='raise':
                self.deal_next_round()
                self.update_game_state(action, next_round=True)
                return

        if self.GameState.current_round > 0:

                # is round complete and its not pre flop?
            if action == 'call' or (self.GameState.previous_action == 'check' and action == 'check'):
                self.deal_next_round()
                self.update_game_state(action,next_round=True)
                return

        # if non of the above the round hasnt ended
        self.update_game_state(action)
        return

    def get_winning_player(self):
        """
        :return: tuple (winning, losing)
        """

        board = self.get_bit_board()

        sb_score = self.evaluator.evaluate(board, self.SB.hand_bit)
        bb_score = self.evaluator.evaluate(board, self.BB.hand_bit)

        if sb_score < bb_score:
            return self.SB,self.BB

        else:
            return self.BB,self.SB

    def init_game(self, SB, BB, GameState):
        """

        :param SB:
        :param BB:
        :param GameState:
        :return:
        """

        self.GameState = GameState

        self.SB = SB

        self.BB = BB

        self.players = ['_',self.SB,self.BB]

        # creating mapping of card bit int to int to make
        # it easier for tracking state
        deck = Deck.GetFullDeck()
        deck_dict = {}
        for i in range(len(deck)):
            deck_dict[deck[i]] = i

        self.deck_lookup = deck_dict

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

        opponent_player = self.players[-1 * self.players.index(current_player)] # I thought this was a cool trick... its not really
        opponent_player_name = opponent_player.name
        opponent_player_stack = opponent_player.stack_size

        current_pot = self.GameState.current_pot

        r = {
            current_player_name: 0,
            opponent_player_name: 0

        }

        if previous_action == 'fold':

            # current player gets the pot
            r[current_player_name] = current_player_stack

            r[opponent_player_name] = current_pot + opponent_player_stack


        elif previous_action == 'call' or previous_action == 'check':

            winning_player, losing_player = self.get_winning_player()

            r[winning_player.name] = current_pot + winning_player.stack_size
            r[losing_player.name] = losing_player.stack_size

        return r

    def run_sim(self, iters):
        """
        :param iters:
        :return:
        """

        for iter in range(iters):

            self.deck.shuffle()

            self.SB.eps = 1.0 / np.sqrt(iter)
            self.BB.eps = 1.0 / np.sqrt(iter)

            self.terminal_state = False

            self.SB.stack_size = self.S
            self.BB.stack_size = self.S

            self.deal()

            print("SB dealt {}".format(Card.print_pretty_cards(self.SB.hand_bit)))
            print("BB dealt {}".format(Card.print_pretty_cards(self.BB.hand_bit)))

            self.GameState.reset(self.SB)

            # post blinds
            self.SB.stack_size -= int(0.5*self.GameState.bb_size)
            self.BB.stack_size -= self.GameState.bb_size

            # set policy
            uniform = np.random.uniform(1)

            if self.anticipatory <= uniform:

                self.SB.current_policy = 'epsilon-greedy'
                self.BB.current_policy = 'epsilon-greedy'

            else:

                self.SB.current_policy = 'policy'
                self.BB.current_policy = 'policy'

            r = self.simulate()
            print("Reward {}".format(str(r)))

            # update networks every ith step

        print("End of game")

    def simulate(self, action=None):
        """
        :param action:
        :return:
        """

        current_state = self.GameState.game_state.copy()

        current_player = self.GameState.current_player.name

        if self.GameState.terminal_state:

            return self.reward(action)

        action_index = self.get_action(current_state)

        action = self.actions[action_index]

        if self.GameState.current_round == 0 and current_player == 'SB' and action == 'call' and self.GameState.num_raises == 0:
            print("LImp sacue")

        print("Player {} takes action {}".format(self.GameState.current_player.name,action))

        self.get_next_state(action)

        self.GameState.previous_action = action

        next_state = self.GameState.game_state.copy()

        print("Current Round {}".format(str(self.GameState.current_round)))
        print("Current Num Raises {}".format(str(self.GameState.num_raises)))

        done = self.GameState.terminal_state

        r = self.simulate(action=action)

        if current_player == 'SB':

            self.SB.step((current_state, action_index, r['SB'], next_state, done))

        elif current_player == 'BB':

            self.BB.step((current_state, action_index, r['BB'], next_state, done))

        return r

    def update_game_state(self,action,next_round=False):
        """

        'raise', 'check', 'call', 'fold'

        :param action:
        :param next_round:
        :return:
        """

        current_player_index = self.players.index(self.GameState.current_player) - 1
        current_round = self.GameState.current_round
        num_raises = self.GameState.num_raises

        if action in ['check','call']:
            action_index = 1

        elif action == 'raise':
            action_index = 0

        else:
            raise Exception("Error action {} was used to update_game_state".format(action))

        self.GameState.game_state[current_player_index][current_round][num_raises][action_index] = 1

        if action in ['raise','call']:

            self.GameState.current_player.stack_size -= self.GameState.bb_size
            self.GameState.current_pot += self.GameState.bb_size

            if action == 'raise':
                self.GameState.num_raises += 1

        if next_round:

            self.GameState.current_round += 1
            self.GameState.current_player = self.BB  # its always the BB turn when the round ends

        else:

            if self.GameState.current_player == self.SB:

                self.GameState.current_player = self.BB
            else:
                self.GameState.current_player = self.SB


class GameState:
    """
    Allow us to be able to pass state between classes
    """

    def __init__(self, game_state, current_player, pot, current_policy='epsilon-greedy'):
        self.bb_size = 2
        self.flop = torch.zeros(52)
        self.bit_flop = None
        self.turn = torch.zeros(52)
        self.bit_turn = None
        self.river = torch.zeros(52)
        self.bit_river = None
        self.current_player = current_player
        self.current_policy = current_policy
        self.current_pot = pot  # starting amount = 1
        self.current_round = 0
        self.game_state = game_state  # tensor that tracks current state
        self.num_raises = 0
        self.previous_action = None
        self.terminal_state = False

    def reset(self, player):
        self.bb_size = 2
        self.flop = torch.zeros(52)
        self.bit_flop = None
        self.turn = torch.zeros(52)
        self.bit_turn = None
        self.river = torch.zeros(52)
        self.bit_river = None
        self.current_player = player
        self.current_pot = self.bb_size + int(0.5*self.bb_size) # bb + sb is the current pot size
        self.current_round = 0
        self.game_state = np.zeros((2, 4, 5, 2))
        self.num_raises = 0
        self.previous_action = None
        self.terminal_state = False

def run_nfsp():

    STATE_SIZE = 288
    ACTION_SIZE = 4
    SEED = 1
    EPISODES = 20
    ITERS = 500

    # Game state { player, round, num raises, action}
    game_state_init = np.zeros((2, 4, 5, 2))

    State = GameState(game_state_init, current_player=None, pot=1.0)

    NFSPGame = Game()

    player1 = PokerAgent(STATE_SIZE, ACTION_SIZE, SEED, State,'SB')

    player2 = PokerAgent(STATE_SIZE, ACTION_SIZE, SEED, State,'BB')

    #sb_opponent = RandomJack(State, NFSPGame.hands, 'SB', 5.0)

    #bb_opponent = RandomJack(State, NFSPGame.hands, 'BB', 5.0)

    hero_sb = []
    hero_bb = []

    for i in range(EPISODES):

        print("Running episode {}".format(str(i)))

        NFSPGame.init_game(player1, player2, State)

        NFSPGame.run_sim(ITERS)

        print("Evaluating SB Agent")


    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(hero_sb)), hero_sb)
    plt.ylabel('Avg R')
    plt.xlabel('x')
    plt.savefig('hero_sb.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(hero_bb)), hero_bb)
    plt.ylabel('Avg R')
    plt.xlabel('x')
    plt.savefig('hero_bb.png')

if __name__ == '__main__':
    run_nfsp()

