import pyspiel
import random
import numpy as np
import math


def epsilon_greedy(state, legal_actions, Q): 
    if random.random() < 0.1: 
        return random.choice(legal_actions)
    max_action, max_Q = None, float('-inf')
    for action in legal_actions: 
        cur_Q = Q.get((str(state), action), 0)
        if cur_Q > max_Q: 
            max_action, max_Q = action, cur_Q
    return max_action 

def calc_scores(state): 
    obs = state.observation_tensor(0)
    your_score, opp_score = sum(obs[1:8]), sum([obs[0]] + obs[8:])
    return your_score, opp_score

def simulate(actions, N_sa, N_s, state, Q):
    if state.is_terminal(): 
        your_score, opp_score = calc_scores(state)
        return your_score - opp_score, 'none'
    action_values = []
    action_idx = []
    for a in actions: 
        UCB = (1 / math.sqrt(2)) 
        Ns = N_s.get(state, 0)
        Nsa = N_sa.get((state, a), 0)
        if Nsa == 0: UCB = float('inf')
        else: UCB *= math.sqrt(math.log(Ns) / Nsa)
        action_values.append(Q.get((state, a), 0) + UCB)
        action_idx.append(a)
    action_values, action_idx = np.array(action_values), np.array(action_idx)
    max_idxs = action_idx[action_values == np.max(action_values)]
    action = random.choice(max_idxs)

    state.apply_action(action)
    q = 0 + 1 * simulate(state.legal_actions(), N_s, N_sa, state, Q)[0] # no reward for now? 
    N_s[state] = N_s.get(state, 0) + 1
    N_sa[(state, action)] = N_sa.get((state, action), 0) + 1
    Q[(state, action)] = (q - Q.get((state, action), 0)) / N_sa.get((state, action), 0) + Q.get((state, action), 0)
    return q, action 

# TODO: make this into class 
def MCTS(Q, N_sa, N_s, prev_game_state): 
    game_scores = []
    actions = []
    for i in range(50): # num simulations of games
        # select action 
        game, state = pyspiel.deserialize_game_and_state(prev_game_state)
        legal_actions = state.legal_actions()
        res_score, res_action = simulate(legal_actions, N_sa, N_s, state, Q)
        game_scores.append(res_score)
        actions.append(res_action) # somehow i am not using the q value lmao? 
    # print(game_scores, actions)
    return actions[np.argmax(game_scores)]




def main(): 
    mancala = pyspiel.load_game("mancala")
    win, total = 0, 10000
    Q, N_sa, N_s = dict(), dict(), dict()
    for i in range(total): 
        state = mancala.new_initial_state()
        while not state.is_terminal(): 
            legal_actions = state.legal_actions()
            # we are player 0 
            if state.current_player() == 0: 
                prev_game_state = pyspiel.serialize_game_and_state(mancala, state)	
                action = MCTS(Q, N_sa, N_s, prev_game_state)
                state.apply_action(action)
            else: # random policy 
                action = random.choice(legal_actions)
                state.apply_action(action)

        final_state = state.observation_tensor(0)
        your_score, opp_score = sum(final_state[1:8]), sum([final_state[0]] + final_state[8:])
        # print('Final Score - ', "You: ", your_score, "Opp: ", opp_score)
        if your_score > opp_score: win += 1
        if i % 20 == 0: print("Win rate at iteration", i + 1, win / (i+1))
    print(win / total)

if __name__ == "__main__":
    main()