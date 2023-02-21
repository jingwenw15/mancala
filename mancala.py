import pyspiel
import random
from MCTS import MCTSAgent
import matplotlib.pyplot as plt 


def epsilon_greedy(state, legal_actions, Q): 
    if random.random() < 0.1: 
        return random.choice(legal_actions)
    max_action, max_Q = None, float('-inf')
    for action in legal_actions: 
        cur_Q = Q.get((str(state), action), 0)
        if cur_Q > max_Q: 
            max_action, max_Q = action, cur_Q
    return max_action 

def solve_MCTS(): 
    MCTS = MCTSAgent()
    mancala = pyspiel.load_game("mancala")
    win, total = 0, 1000
    acc_y, num_games_x = [], []
    for i in range(1, total+1): 
        state = mancala.new_initial_state()
        while not state.is_terminal(): 
            legal_actions = state.legal_actions()
            # we are player 0 
            if state.current_player() == 0: 
                prev_game_state = pyspiel.serialize_game_and_state(mancala, state)	
                action = MCTS.MCTS(prev_game_state)
                state.apply_action(action)
            else: # random policy 
                action = random.choice(legal_actions)
                state.apply_action(action)

        final_state = state.observation_tensor(0)
        your_score, opp_score = sum(final_state[1:8]), sum([final_state[0]] + final_state[8:])
        if your_score > opp_score: win += 1
        if (i) % 20 == 0: print("Win rate at iteration", i, win / (i))
        # add to plot 
        acc_y.append(win / i)
        num_games_x.append(i)

    print(win / total)
    plt.plot(num_games_x, acc_y, '-')
    ax = plt.gca()
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Percentage of wins")
    ax.set_xlabel("Number of games played")
    ax.set_title("MCTS Agent vs Random Policy")
    plt.show()


def main(): 
    solve_MCTS()

if __name__ == "__main__":
    main()