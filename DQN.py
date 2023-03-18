from stable_baselines3 import DQN, dqn, a2c, A2C
from stable_baselines3.common.env_checker import check_env
from MancalaEnv import MancalaEnv
# from MancalaEnvSecondPlayer import MancalaEnv
# from MancalaEnvFirstPlayer import MancalaEnv
import matplotlib.pyplot as plt 
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
dqnpolicy = dqn.MlpPolicy
# model = DQN(policy, env, verbose=1)
# model.learn(total_timesteps=10000000, log_interval=1000)
# model.save("DeepQ313")

# model.set_parameters("DQN")
model = DQN(dqnpolicy, env, verbose=1)
model.set_parameters("DQN")
model2 = DQN(dqnpolicy, env, verbose=1)
model2.set_parameters("DQN2")
model3 = DQN(dqnpolicy, env, verbose=1)
model3.set_parameters("DQN3")
print('Loaded DQN models')
# env = model.get_env()
# obs = env.reset()

a2cpolicy = a2c.MlpPolicy

model4 = A2C(a2cpolicy, env, verbose=1)
model4.set_parameters("A2C")

model5 = A2C(a2cpolicy, env, verbose=1)
model5.set_parameters("A2C2")

model6 = A2C(a2cpolicy, env, verbose=1)
model6.set_parameters("A2C3")
print("Loaded A2C Models")

MCTS = MCTSAgent(25, player_one=False)

# initialize games
mancala = pyspiel.load_game("mancala")
num_games = 1000

# TODO: Idea what if we just dont count unsuccessful runs? 
def play_DQN(player_2): 
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

            else: # opponent
                opp_obs = [num for num in state.observation_tensor(0)]
                opp_obs = opp_obs[7:] + opp_obs[:7]
                action = None 
                if player_2 == "Random": 
                    action = random.choice(state.legal_actions()) - 8 
                if player_2 == "DQN": 
                    action1, _ = model.predict(opp_obs, deterministic=True)
                    action2, _ = model2.predict(opp_obs, deterministic=True)
                    action3, _ = model3.predict(opp_obs, deterministic=True)
                    action = random.choice([action1, action2, action3])
                elif player_2 == "A2C": 
                    action4, _ = model4.predict(opp_obs, deterministic=True)
                    action5, _ = model5.predict(opp_obs, deterministic=True)
                    action6, _ = model6.predict(opp_obs, deterministic=True)
                    action = random.choice([action4, action5, action6])
                elif player_2 == "MCTS": 
                    prev_game_state = pyspiel.serialize_game_and_state(mancala, state)	
                    action = MCTS.MCTS(prev_game_state) - 8
                elif player_2 == "Q-Learning": 
                    action = QLAgent.make_move(state)
                    continue # QL Agent applies the move in make_move... bad design, I know but sunk cost 

                if action + 8 in state.legal_actions(): # pyspiel has 1 indexing for actions
                    state.apply_action(action + 8)
                else: # illegal action taken
                    success = False 
                    break
                # QLearning.make_move(state)
        if success: 
            p1_score, p2_score = calc_scores(state)
            if p1_score > p2_score: 
                total_wins += 1
            elif p1_score < p2_score:
                opp_wins += 1 
            total_games += 1 
            if total_games % 100 == 0: print("Completed game", total_games)
        else: 
            fails += 1 
        if total_games == 1000: 
            break 

    print(total_wins / total_games)
    print("opp wins", opp_wins / total_games)
    print("fails", fails)

play_DQN("Q-Learning")

# plt.plot(iteration_num, win_rate, '-')
# ax = plt.gca()
# ax.set_ylim([0, 1.1])
# ax.set_ylabel("Win rate over past 100 games")
# ax.set_xlabel("Number of games played")
# ax.set_title("DQN Agent vs Random Policy: DQN as Player 1")
# plt.savefig('DQN Agent vs Random Policy: DQN as Player 1')