from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_checker import check_env
from MancalaEnv import MancalaEnv


env = MancalaEnv()
check_env(env)
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000000, log_interval=1000)
model.save("A2C")

model = A2C.load("A2C", env)

env = model.get_env()
obs = env.reset()
wins = 0
total_games = 0
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done and reward != -1000: 
        # an actual game has been played out 
        env.reset()
        if reward > 0: wins += 1
        total_games += 1
print(wins/total_games)
