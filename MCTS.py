import random
import math 
import numpy as np
import copy
import pyspiel

class MCTSAgent(): 
    def __init__(self):
        self.Q = dict()
        self.N_sa = dict()
        self.N_s = dict()

    def calc_scores(self, state): 
        obs = state.observation_tensor(0)
        your_score, opp_score = sum(obs[1:8]), sum([obs[0]] + obs[8:])
        return your_score, opp_score
    
    def simulate(self, actions, state):
        if state.is_terminal(): 
            your_score, opp_score = self.calc_scores(state)
            return your_score - opp_score, 'none'
        action_values = []
        action_idx = []
        for a in actions: 
            UCB = (1 / math.sqrt(2)) 
            Ns = self.N_s.get(state, 0)
            Nsa = self.N_sa.get((state, a), 0)
            if Nsa == 0: UCB = float('inf')
            else: UCB *= math.sqrt(math.log(Ns) / Nsa)
            action_values.append(self.Q.get((state, a), 0) + UCB)
            action_idx.append(a)
        action_values, action_idx = np.array(action_values), np.array(action_idx)
        max_idxs = action_idx[action_values == np.max(action_values)]
        action = random.choice(max_idxs)

        # save prev state
        prev_state = str(copy.copy(state))
        state.apply_action(action)

        q = 0 + 1 * self.simulate(state.legal_actions(), state)[0] # no reward for now until terminal state
        self.N_s[prev_state] = self.N_s.get(prev_state, 0) + 1
        self.N_sa[(prev_state, action)] = self.N_sa.get((prev_state, action), 0) + 1
        self.Q[(prev_state, action)] = (q - self.Q.get((prev_state, action), 0)) / self.N_sa.get((prev_state, action), 0) + self.Q.get((prev_state, action), 0)
        return q, action

    def MCTS(self, prev_game_state): 
        actions = []
        for i in range(50): # num simulations of games
            # select action
            game, state = pyspiel.deserialize_game_and_state(prev_game_state)
            legal_actions = state.legal_actions()
            self.simulate(legal_actions, state)

        game, state = pyspiel.deserialize_game_and_state(prev_game_state)
        action_scores, actions = [], []

        for action in state.legal_actions(): 
            actions.append(action)
            action_scores.append(self.Q.get((str(state), action), 0))
        

        return actions[np.argmax(action_scores)]