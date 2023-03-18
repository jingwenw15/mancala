import pyspiel
import random
from MCTS import MCTSAgent
from QLearning import QLearningAgent
import matplotlib.pyplot as plt 
from MancalaEnv import MancalaEnv
from stable_baselines3 import DQN, dqn, a2c, A2C


def play_MCTS(player_two): 
    MCTSPlayer1 = MCTSAgent(25, player_one=True)
    MCTSPlayer2 = MCTSAgent(25, player_one=False)
    QLearning = QLearningAgent(player_one=False) # Q Learning as second player 
    if player_two == "Q-Learning": QLearning.QLearning(load_data=True)
    mancala = pyspiel.load_game("mancala")

    env = MancalaEnv()
    model, model2, model3 = None, None, None 
    if player_two == "DQN": 
        dqnpolicy = dqn.MlpPolicy
        model = DQN(dqnpolicy, env, verbose=1)
        model.set_parameters("DQN")
        model2 = DQN(dqnpolicy, env, verbose=1)
        model2.set_parameters("DQN2")
        model3 = DQN(dqnpolicy, env, verbose=1)
        model3.set_parameters("DQN3")
        print('Loaded DQN models')
    elif player_two == "A2C": 
        a2cpolicy = a2c.MlpPolicy
        model = A2C(a2cpolicy, env, verbose=1)
        model.set_parameters("A2C")
        model2 = A2C(a2cpolicy, env, verbose=1)
        model2.set_parameters("A2C2")
        model3 = A2C(a2cpolicy, env, verbose=1)
        model3.set_parameters("A2C3")
        print('Loaded A2C models')

    wins, total = 0, 1000
    total_wins = 0
    iteration_num, win_rate = [], []
    opp_wins = 0 
    num_games = 0 
    while True: 
        success = True
        state = mancala.new_initial_state()
        while not state.is_terminal(): 
            legal_actions = state.legal_actions()
            if state.current_player() == 0:
                prev_game_state = pyspiel.serialize_game_and_state(mancala, state)	
                action = MCTSPlayer1.MCTS(prev_game_state)
                state.apply_action(action)
            else: # opponent
                if player_two == "Q-Learning": 
                    QLearning.make_move(state)
                elif player_two == "Random": 
                    action = random.choice(legal_actions)
                    state.apply_action(action)
                elif player_two == "MCTS": 
                    prev_game_state = pyspiel.serialize_game_and_state(mancala, state)	
                    action = MCTSPlayer2.MCTS(prev_game_state)
                    state.apply_action(action)
                elif player_two == "A2C" or player_two == "DQN": 
                    opp_obs = [num for num in state.observation_tensor(0)]
                    opp_obs = opp_obs[7:] + opp_obs[:7]
                    action, _ = model.predict(opp_obs, deterministic=True) 
                    action += 8
                    # action2, _ = model2.predict(opp_obs, deterministic=True)
                    # action3, _ = model3.predict(opp_obs, deterministic=True)
                    # action = random.choice([action1, action2, action3]) + 8 
                    if action in state.legal_actions():
                        state.apply_action(action)
                    else: 
                        success = False 
                        break  
        if success == True: 
            num_games += 1
            your_score, opp_score = MCTSPlayer1.calc_scores(state)
            if your_score > opp_score:
                total_wins += 1
                wins += 1
            elif opp_score > your_score:
                opp_wins += 1
            if num_games % 100 == 0: 
                iteration_num.append(num_games)
                win_rate.append(wins / 100)
                print("Win rate for past 100 games at iteration", num_games, wins / 100)
                wins = 0
            if num_games == 1000: break 
    print("Total win rate out of 1000 games", total_wins / total)
    print("Opp win rate out of 1000 games", opp_wins / total)

    # plt.plot(iteration_num, win_rate, '-')
    # ax = plt.gca()
    # ax.set_ylim([0, 1.1])
    # ax.set_ylabel("Win rate over past 100 games")
    # ax.set_xlabel("Number of games played")
    # ax.set_title("MCTS Agent vs Random Policy: Agent as Player 2")
    # plt.savefig('MCTS Agent vs Random Policy: Agent as Player 2')

def play_Random(player_2): 
    mancala = pyspiel.load_game("mancala")
    def calc_scores(state): 
        obs = state.observation_tensor(0)
        player_1_score, player_2_score = sum(obs[1:8]), sum([obs[0]] + obs[8:])
        return player_1_score, player_2_score
    wins, total = 0, 1000
    total_wins = 0
    opp_wins = 0 
    iteration_num, win_rate = [], []
    QLAgent = None 
    if player_2 == "Q-Learning": 
        QLAgent = QLearningAgent()
        QLAgent.QLearning(load_data=True)
    for i in range(1, total+1): 
        state = mancala.new_initial_state()
        while not state.is_terminal():
            legal_actions = state.legal_actions()
            if state.current_player() == 0:
                action = random.choice(legal_actions)
                state.apply_action(action)
            else: 
                action = None 
                if player_2 == "Random": 
                    action = random.choice(legal_actions)
                    state.apply_action(action)
                elif player_2 == "Q-Learning": 
                    action = QLAgent.make_move(state)

        your_score, opp_score = calc_scores(state)
        if your_score > opp_score: 
            wins += 1
            total_wins += 1
        elif opp_score > your_score: 
            opp_wins += 1
        if i % 100 == 0: 
            print("Win rate at iteration", i, wins / 100)
            iteration_num.append(i)
            win_rate.append(wins / 100)
            wins = 0
    print("Total win rate out of 1000 games", total_wins / total)
    print("Second player number of wins", opp_wins / total)

def solve_Q_Learning(): 
    QLearning = QLearningAgent()
    QLearning.QLearning(load_data=False)
    # QLearning.QLearning(load_data=True)
    QLearning.save_Q_dict('QL317.txt')
    QLearning.play_games(player_one=True)
    QLearning.play_games(player_one=False)

def play_Q_Learning(player_two): 
    mancala = pyspiel.load_game("mancala")
    wins = 0 
    total_wins = 0 
    opp_wins = 0 
    iteration_num, win_rate = [], []
    num_games = 1000

    QLAgent = QLearningAgent()
    QLAgent.QLearning(load_data=True,  data_path="QLearning312.txt")
    QLAgent2 = QLearningAgent(player_one=False) 
    QLAgent3 = QLearningAgent()
    QLAgent3.QLearning(load_data=True, data_path="QLearning313.txt")

    env = MancalaEnv()
    model, model2, model3 = None, None, None 
    if player_two == "DQN": 
        dqnpolicy = dqn.MlpPolicy
        model = DQN(dqnpolicy, env, verbose=1)
        model.set_parameters("DQN")
        model2 = DQN(dqnpolicy, env, verbose=1)
        model2.set_parameters("DQN2")
        model3 = DQN(dqnpolicy, env, verbose=1)
        model3.set_parameters("DQN3")
        print('Loaded DQN models')
    elif player_two == "A2C": 
        a2cpolicy = a2c.MlpPolicy
        model = A2C(a2cpolicy, env, verbose=1)
        model.set_parameters("A2C")
        model2 = A2C(a2cpolicy, env, verbose=1)
        model2.set_parameters("A2C2")
        model3 = A2C(a2cpolicy, env, verbose=1)
        model3.set_parameters("A2C3")
        print('Loaded A2C models')

    if player_two == "Q-Learning": QLAgent2.QLearning(load_data=True, data_path="QLearning313.txt")
    print("Loaded Q Learning agents")
    MCTS = MCTSAgent(25, player_one=False)

    num_games = 0 
    while True: 
        state = mancala.new_initial_state()
        success = True 
        while not state.is_terminal(): 
            if state.current_player() == 0: 
                if random.random() < 0.5: QLAgent.make_move(state)
                else: QLAgent3.make_move(state)
            else: # opponent 
                action = None
                if player_two == "MCTS": 
                    prev_game_state = pyspiel.serialize_game_and_state(mancala, state)	
                    action = MCTS.MCTS(prev_game_state) 
                    state.apply_action(action)
                elif player_two == "Q-Learning": 
                    if random.random() < 0.5: QLAgent.make_move(state)
                    else: QLAgent2.make_move(state)
                    continue # because make move actually makes the move 
                elif player_two == "DQN" or player_two == "A2C": 
                    opp_obs = [num for num in state.observation_tensor(0)]
                    opp_obs = opp_obs[7:] + opp_obs[:7]
                    action1, _ = model.predict(opp_obs, deterministic=True)
                    action2, _ = model2.predict(opp_obs, deterministic=True)
                    action3, _ = model3.predict(opp_obs, deterministic=True)
                    action = random.choice([action1, action2, action3]) + 8 
                    if action in state.legal_actions():
                        state.apply_action(action)
                    else: 
                        success = False 
                        break  
        if success == True: 
            num_games += 1
            your_score, opp_score = QLAgent.calc_scores(state)
            if your_score > opp_score:
                total_wins += 1
                wins += 1
            elif opp_score > your_score:
                opp_wins += 1
            if num_games % 100 == 0: 
                iteration_num.append(num_games)
                win_rate.append(wins / 100)
                print("Win rate for past 100 games at iteration", num_games, wins / 100)
                wins = 0
            if num_games == 1000: break 
    print('total wins', total_wins / num_games)

def plot_Q_Training(): 
    QLearning = QLearningAgent()
    QLearning.plot_training()

def main(): 
    # play_MCTS('A2C')
    # solve_Q_Learning()
    # play_Random('Q-Learning')
    # play_Q_Learning("A2C")
    plot_Q_Training()

if __name__ == "__main__":
    main()