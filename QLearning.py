import random
import math 
import numpy as np
import copy
import pyspiel
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt 
import csv

class QLearningAgent(): 
    def __init__(self, player_one=True):
        self.Q = defaultdict(int)
        self.alpha = 0.4
        self.gamma = 0.9
        self.num_sims = 2500000
        self.num_games = 1000
        self.player_one = player_one 

    def calc_scores(self, state): 
        obs = state.observation_tensor(0)
        player_1_score, player_2_score = sum(obs[1:8]), sum([obs[0]] + obs[8:])
        if self.player_one: return player_1_score, player_2_score  
        return player_2_score, player_1_score
    
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
    
    def plot_graph(self, iteration_num, win_rate, title, x_label, y_label, line_label): 
        color = 'pink' if line_label == 'Agent as player 1' else 'lightblue'
        plt.plot(iteration_num, win_rate, color, label=line_label)
        ax = plt.gca()
        ax.set_ylim([0.4, 0.8])
        plt.rcParams["font.family"] = "sans serif"
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        plt.legend(loc='upper left')
        plt.savefig(title + '.png')

    # to obtain good estimate of value function 
    def QLearning(self, load_data=False, data_path='QLearning313.txt'): 
        if load_data: 
            file = open(data_path, 'rb')
            self.Q = pickle.load(file)
            file.close() 
            print("Loaded Q state dict")
            return 
        train_file = open('QL_Training.csv', 'w')
        writer = csv.writer(train_file)

        for player_num in range(2): 
            iteration_num, win_rate = [], [] # for graphing 
            mancala = pyspiel.load_game("mancala")
            wins = 0
            self.player_one = True if player_num == 0 else False
            for game in range(1, self.num_sims + 1): 
                state = mancala.new_initial_state()
                s, a = None, None
                last_update = False 
                while True: 
                    if state.current_player() == player_num or last_update:
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
                if game % 1000 == 0:             
                    iteration_num.append(game)
                    win_rate.append(wins / 1000)
                    writer.writerow([game, win_rate])
                    print("Win rate in past 1000 iterations at iteration", game, wins / 1000)
                    wins = 0 

            self.plot_graph(iteration_num, win_rate, 'Q-Learning Training vs Random Policy', \
                            x_label='Number of iterations', y_label="Win rate in past 1000 games", line_label="Agent as player " + str(player_num + 1))
        
    def save_Q_dict(self, fileName): 
        file = open(fileName, 'wb')
        pickle.dump(self.Q, file)
        file.close()
    
    def make_move(self, state): 
        actions, action_Q_vals = [], []
        for action in state.legal_actions(): 
            cur_Q = self.Q[(str(state), action)]
            actions.append(action)
            action_Q_vals.append(cur_Q)
        actions = np.array(actions)
        state.apply_action(random.choice(list(actions[abs(action_Q_vals - np.max(action_Q_vals)) < 1e-6]))) # choose random action if tied

    def play_games(self, player_one=True): 
        mancala = pyspiel.load_game("mancala")
        wins = 0 
        total_wins = 0 
        opp_wins = 0 
        iteration_num, win_rate = [], []
        player_num = 0 if player_one == True else 1 
        self.player_one = player_one 
        for game in range(1, self.num_games + 1): 
            state = mancala.new_initial_state()
            while not state.is_terminal(): 
                if state.current_player() == player_num: 
                    self.make_move(state)
                else: # opponent 
                    actn = self.random_agent(state)
                    state.apply_action(actn)
            your_score, opp_score = self.calc_scores(state)
            if your_score > opp_score:
                total_wins += 1
                wins += 1
            elif opp_score > your_score:
                opp_wins += 1
            if game % 100 == 0: 
                iteration_num.append(game)
                win_rate.append(wins / 100)
                print("Win rate for past 100 games at iteration", game, wins / 100)
                wins = 0
        self.plot_graph(iteration_num, win_rate, 'Q-Learning vs Random Policy',
                        x_label="Number of Iterations", y_label="Win rate for past 100 games", line_label='Agent as Player ' + str(player_num + 1))

        print(total_wins / self.num_games)
        print("opp wins", opp_wins / self.num_games)
    
    def plot_training(self): 
        f = open('QL_Training.csv', 'r')
        i = 1
        for line in f: 
            split = line.split('","')
            iteration_num, win_rate = split[0], split[1]

            iteration_num = iteration_num[2:len(iteration_num) - 1]
            iteration_num = iteration_num.split(', ')
            iteration_num = [int(n) for n in iteration_num]

            win_rate = win_rate[2:len(win_rate) - 3]
            win_rate = win_rate.split(', ')
            win_rate = [float(n) for n in win_rate]
            
            self.plot_graph(iteration_num, win_rate, 'Q-Learning Training vs Random Policy', \
                            x_label='Number of iterations', y_label="Win rate in past 1000 games", line_label="Agent as player " + str(i))
            i += 1
