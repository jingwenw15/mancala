import gym
import numpy as np
from gym import spaces
import pyspiel
import random

class MancalaEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.MultiDiscrete([48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48])
        self.mancala = pyspiel.load_game("mancala")
        self.pyspiel_state = self.mancala.new_initial_state() 
        
    def step(self, action):
        # return valid_actions array of 1s and 0s in info dict 
        reward = 0 
        if self.pyspiel_state.current_player() == 0: 
            if action + 1 in self.pyspiel_state.legal_actions(): 
                self.pyspiel_state.apply_action(action + 1)
            else: # illegal action taken 
                reward = -1000
                return np.array([num for num in self.pyspiel_state.observation_tensor(0)]), reward, True, {}

        # take opponent step
        while self.pyspiel_state.current_player() == 1 and not self.pyspiel_state.is_terminal(): 
            action = random.choice(self.pyspiel_state.legal_actions())
            self.pyspiel_state.apply_action(action)
            
        if self.pyspiel_state.is_terminal(): 
            your_score, opp_score = self.calc_scores(self.pyspiel_state)
            reward = your_score - opp_score

        obs_array = np.array([num for num in self.pyspiel_state.observation_tensor(0)])
        return obs_array, reward, self.pyspiel_state.is_terminal(), {}

    def calc_scores(self, state): 
        obs = state.observation_tensor(0)
        your_score, opp_score = sum(obs[1:8]), sum([obs[0]] + obs[8:])
        return your_score, opp_score
    
    def reset(self):
        self.pyspiel_state = self.mancala.new_initial_state() 
        return np.array(self.pyspiel_state.observation_tensor(0)) # reward, done, info can't be included
    
    def render(self, mode='human'): 
        return str(self.pyspiel_state)

