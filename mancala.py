import pyspiel
import random
import numpy as np


def epsilon_greedy(state, legal_actions, Q): 
    if random.random() < 0.1: 
        return random.choice(legal_actions)
    max_action, max_Q = None, float('-inf')
    for action in legal_actions: 
        cur_Q = Q.get((str(state), action), 0)
        if cur_Q > max_Q: 
            max_action, max_Q = action, cur_Q
    return max_action 

def sarsa(state, obs, legal_actions, Q, prev_R, prev_S, prev_A, alpha=0.9, discount=0.9, first_time=False): 
    action = epsilon_greedy(state, legal_actions, Q)
    if not first_time: 
        prev_Q = Q.get((str(prev_S), prev_A), 0)
        Q[(str(prev_S), prev_A)] = prev_Q + alpha * (prev_R + discount * Q.get((str(obs), action), 0) - prev_Q)
    return action, obs

def reward(before_S: list, after_S: list, state): 
    turn_bonus = 10 if state.current_player() == 0 else 0 
    my_score_after, opp_score_after = after_S[7], after_S[0]
    my_score_before, opp_score_before = before_S[7], before_S[0]
    capture_bonus = my_score_after - my_score_before 
    my_marbles_before, opp_marbles_before = sum(before_S[1:7]), sum(before_S[8:])
    my_marbles_after, opp_marbles_after = sum(after_S[1:7]), sum(after_S[8:])
    marble_diff_reward = 0.5 * ((my_marbles_after - opp_marbles_after) - (my_marbles_before - opp_marbles_before))
    outcome_bonus = 0.5 * (my_score_after - opp_score_after)
    win_bonus = 10 if state.is_terminal() else -10
    return turn_bonus + capture_bonus + win_bonus + marble_diff_reward + outcome_bonus

def main(): 
    # TODO: SARSA modification idea - store only simpler state info like marbles on each side, total marbles, to reduce state space
    mancala = pyspiel.load_game("mancala")
    Q = dict()
    win, total = 0, 1000000
    for i in range(total): 
        state = mancala.new_initial_state()
        prev_S, prev_A, prev_R = None, None, None 
        while not state.is_terminal(): 
            legal_actions = state.legal_actions()
            # we are player 0 
            if state.current_player() == 0: 
                first_time = not (prev_S and prev_A and prev_R)
                obs = state.observation_tensor() 
                if not first_time: prev_R = reward(prev_S, state.observation_tensor(), state)
                prev_A, prev_S = sarsa(state, obs, legal_actions, Q, prev_R, prev_S, prev_A, first_time=first_time)
                state.apply_action(prev_A)

            else: # random policy 
                action = random.choice(legal_actions)
                state.apply_action(action)
        # last sarsa update 
        final_state = state.observation_tensor(0)   
        prev_R = reward(prev_S, final_state, state) 
        Q[(str(prev_S), prev_A)] = Q.get((str(prev_S), prev_A), 0) + 0.9 * (prev_R - Q.get((str(prev_S), prev_A), 0))

        your_score, opp_score = sum(final_state[1:8]), sum([final_state[0]] + final_state[8:])
        # print('Final Score - ', "You: ", your_score, "Opp: ", opp_score)
        if your_score > opp_score: win += 1
    print(win / total)
    # print(Q)

if __name__ == "__main__":
    main()