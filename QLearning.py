import random
import math 
import numpy as np
import copy
import pyspiel
from collections import defaultdict
from scipy.interpolate import NearestNDInterpolator
import pickle
import matplotlib.pyplot as plt 

class QLearningAgent(): 
    def __init__(self):
        # self.Q = np.zeros((48, 48, 6))
        self.Q = defaultdict(int)
        self.alpha = 0.4
        self.gamma = 0.9
        self.num_sims = 1000000
        self.num_games = 100000

    def calc_scores(self, state): 
        obs = state.observation_tensor(0)
        your_score, opp_score = sum(obs[1:8]), sum([obs[0]] + obs[8:])
        return your_score, opp_score
    
    def epsilon_greedy(self, state, legal_actions): 
        if random.random() < 0.1: 
            return random.choice(legal_actions)
        actions, action_Q_vals = [], []
        for action in legal_actions: 
            cur_Q = self.Q[(str(state), action)]
            actions.append(action)
            action_Q_vals.append(cur_Q)

        actions = np.array(actions)
        
        return random.choice(list(actions[abs(action_Q_vals - np.max(action_Q_vals)) < 1e-6])) # choose random action if tied 
    
    def select_action(self, state, legal_actions): 
        return self.epsilon_greedy(state, legal_actions)
    
    def random_agent(self, s): 
        return random.choice(s.legal_actions())

    # to obtain good estimate of value function 
    def QLearning(self, load_data=False): 
        if load_data: 
            file = open('QLearningAVF.txt', 'rb')
            self.Q = pickle.load(file)
            file.close() 
            return 
        
        mancala = pyspiel.load_game("mancala")
        wins = 0
        for game in range(1, self.num_sims + 1): 
            state = mancala.new_initial_state()
            s, a = None, None
            last_update = False 
            while True: 
                # we are player 0 
                if state.current_player() == 0 or last_update:
                    if s and a: 
                        # not the first turn, can update 
                        max_Q_sp = float('-inf') 
                        if last_update or not state.legal_actions(): 
                            max_Q_sp = 0 
                        else: 
                            for possible_a in state.legal_actions(): # state = sp; s, a = old s and old a 
                                if self.Q[(str(state), possible_a)] > max_Q_sp: 
                                    max_Q_sp = self.Q[(str(state), possible_a)]
                        r = 0 
                        yr_prev_score, opp_prev_score = self.calc_scores(s)
                        your_score, opp_score = self.calc_scores(state)
                        if last_update or state.is_terminal(): 
                            r = your_score - opp_score
                        else: 
                            r = 0.3 * (your_score - yr_prev_score)
                        assert str(s) != str(state)

                        self.Q[(str(s), a)] += self.alpha * (r + self.gamma * max_Q_sp - self.Q[(str(s), a)])

                    if not state.is_terminal() and not last_update and state.legal_actions: 
                        ap = self.select_action(state, state.legal_actions())
                        s = copy.copy(state)
                        a = ap
                        state.apply_action(ap)
                    else: break 
                else: 
                    if not state.is_terminal(): 
                        random_action = self.random_agent(state)
                        state.apply_action(random_action)
                    else: 
                        last_update = True
            # game finished 
            your_score, opp_score = self.calc_scores(state)
            if your_score > opp_score: wins += 1
            if game % 500 == 0: 
                print("Win rate in past 500 iterations at iteration", game, wins / 500)
                wins = 0 
        
        file = open('QLearningAVF.txt', 'wb')
        pickle.dump(self.Q, file)
        file.close()

    def play_games(self): 
        mancala = pyspiel.load_game("mancala")
        wins = 0 
        iteration_num, win_rate = [], []
        for game in range(1, self.num_games + 1): 
            state = mancala.new_initial_state()
            while not state.is_terminal(): 
                if state.current_player == 0: 
                    actions, action_Q_vals = [], []
                    for action in state.legal_actions(): 
                        cur_Q = self.Q[(str(state), action)]
                        actions.append(action)
                        action_Q_vals.append(cur_Q)
                    actions = np.array(actions)
                    return random.choice(list(actions[abs(action_Q_vals - np.max(action_Q_vals)) < 1e-6])) # choose random action if tied 
                else: 
                    actn = self.random_agent(state)
                    state.apply_action(actn)
            your_score, opp_score = self.calc_scores(state)
            if your_score > opp_score: wins += 1
            if game % 100 == 0: 
                iteration_num.append(game)
                win_rate.append(wins / 100)
                print("Win rate for past 100 games at iteration", game, wins / 100)
                wins = 0
        
        plt.plot(iteration_num, win_rate, '-')
        ax = plt.gca()
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("Win rate over past 100 games")
        ax.set_xlabel("Number of games played")
        ax.set_title("Q Learning Agent vs Random Policy")
        plt.show()


