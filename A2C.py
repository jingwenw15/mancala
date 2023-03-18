from stable_baselines3 import a2c, A2C, DQN, dqn, common
from stable_baselines3.common.env_checker import check_env
from MancalaEnv import MancalaEnv
import numpy as np
import pyspiel
import random 
from MCTS import MCTSAgent
from QLearning import QLearningAgent

def calc_scores(state): 
    obs = state.observation_tensor(0)
    player_1_score, player_2_score = sum(obs[1:8]), sum([obs[0]] + obs[8:])
    return player_1_score, player_2_score  

env = MancalaEnv()
check_env(env)
a2cpolicy = a2c.MlpPolicy

# Load player 1 
model = A2C(a2cpolicy, env, verbose=1)
model.set_parameters("A2C")

model2 = A2C(a2cpolicy, env, verbose=1)
model2.set_parameters("A2C2")

model3 = A2C(a2cpolicy, env, verbose=1)
model3.set_parameters("A2C3")

# Load player 2 
dqnpolicy = dqn.MlpPolicy

model4 = DQN(dqnpolicy, env, verbose=1)
model4.set_parameters("DQN")

model5 = DQN(dqnpolicy, env, verbose=1)
model5.set_parameters("DQN2")

model6 = DQN(dqnpolicy, env, verbose=1)
model6.set_parameters("DQN3")

MCTS = MCTSAgent(25, player_one=False)

# env = model.get_env()
# obs = env.reset()

# initialize games
mancala = pyspiel.load_game("mancala")
num_games = 1000

def play_A2C(player_2): 
    QLAgent = None 
    if player_2 == "Q-Learning": 
        QLAgent = QLearningAgent()
        QLAgent.QLearning(load_data=True)

    iteration_num, win_rate = [], []
    wins, total_games, fails, total_wins, opp_wins = 0, 0, 0, 0, 0
    while True: 
        state = mancala.new_initial_state()
        success = True
        while not state.is_terminal(): 
            legal_actions = state.legal_actions()
            if state.current_player() == 0: 
                action1, _ = model.predict(state.observation_tensor(0), deterministic=True)
                action2, _ = model2.predict(state.observation_tensor(0), deterministic=True)
                action3, _ = model3.predict(state.observation_tensor(0), deterministic=True)
                action = random.choice([action1, action2, action3])
                if action + 1 in state.legal_actions(): 
                    state.apply_action(action + 1)
                else: # illegal action taken 
                    success = False 
                    break
            # take opponent step
            elif state.current_player() == 1: 
                opp_obs = [num for num in state.observation_tensor(0)]
                opp_obs = opp_obs[7:] + opp_obs[:7]
                action = None 
                if player_2 == "DQN": 
                    action1, _ = model4.predict(np.array(opp_obs), deterministic=True)
                    action2, _ = model5.predict(np.array(opp_obs), deterministic=True)
                    action3, _ = model6.predict(np.array(opp_obs), deterministic=True)
                    action = random.choice([action1, action2, action3])
                elif player_2 == "Random": 
                    action = random.choice(state.legal_actions()) - 1 
                elif player_2 == "MCTS": 
                    prev_game_state = pyspiel.serialize_game_and_state(mancala, state)	
                    action = MCTS.MCTS(prev_game_state) - 8
                elif player_2 == "Q-Learning": 
                    action = QLAgent.make_move(state)
                    continue # QL Agent applies the move in make_move... bad design, I know but sunk cost 

                if action + 8 in state.legal_actions(): 
                    state.apply_action(action + 8)
                else: # illegal action taken 
                    success = False 
                    break
        if success: 
            p1_score, p2_score = calc_scores(state)
            if p1_score > p2_score: 
                total_wins += 1
            elif p1_score < p2_score:
                opp_wins += 1 
            # print(p1_score, p2_score)
            total_games += 1 
            if total_games % 100 == 0: print("Completed game", total_games)
        else: 
            fails += 1
        if total_games == num_games: 
            break
    print(total_wins / total_games)
    print("opp wins", opp_wins / total_games)
    print("fails", fails)

play_A2C('MCTS')
# plt.plot(iteration_num, win_rate, '-')
# ax = plt.gca()
# ax.set_ylim([0, 1.1])
# ax.set_ylabel("Win rate over past 100 games")
# ax.set_xlabel("Number of games played")
# ax.set_title("DQN Agent vs Random Policy: DQN as Player 1")
# plt.savefig('DQN Agent vs Random Policy: DQN as Player 1')