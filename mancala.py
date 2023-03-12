import pyspiel
import random
from MCTS import MCTSAgent
from QLearning import QLearningAgent
import matplotlib.pyplot as plt 


def solve_MCTS(): 
    MCTS = MCTSAgent(50)
    mancala = pyspiel.load_game("mancala")
    wins, total = 0, 1
    iteration_num, win_rate = [], []
    for i in range(1, total+1): 
        state = mancala.new_initial_state()
        while not state.is_terminal(): 
            print(state)
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
        if your_score > opp_score: wins += 1
        if i % 100 == 0: 
            print("Win rate at iteration", i, wins / 100)
            iteration_num.append(i)
            win_rate.append(wins / 100)
            wins = 0


    plt.plot(iteration_num, win_rate, '-')
    ax = plt.gca()
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Win rate over past 100 games")
    ax.set_xlabel("Number of games played")
    ax.set_title("MCTS Agent vs Random Policy")
    plt.show()

def solve_Q_Learning(): 
    QLearning = QLearningAgent()
    QLearning.QLearning()
    QLearning.play_games()

def main(): 
    solve_MCTS()
    # solve_Q_Learning()

if __name__ == "__main__":
    main()