from stable_baselines3 import A2C, a2c
from stable_baselines3.common.env_checker import check_env
from MancalaEnv import MancalaEnv
import numpy as np

env = MancalaEnv()
check_env(env)
policy = a2c.MlpPolicy
model = A2C(policy, env, verbose=1, tensorboard_log='runs')
model.learn(total_timesteps=1000000, log_interval=1000)
model.save("A2C3")

# model.set_parameters("A2C")

env = model.get_env()
obs = env.reset()
wins = 0
total_games = 0
fails = 0
for i in range(20000):
    # action, _states = model.predict(obs, deterministic=True)
    obs_t = model.policy.obs_to_tensor(np.array([num for num in obs]))[0]
    prob_dist = model.policy.get_distribution(obs_t).distribution.probs.detach().numpy()[0]
    action = np.argmax(prob_dist)
    assert action == model.predict(obs, deterministic=True)[0]
    obs, reward, done, info = env.step([action])
    if done and reward != -1000: 
        # an actual game has been played out 
        obs = env.reset()
        if reward > 0: wins += 1
        total_games += 1
    elif done: 
        fails += 1
print(wins/total_games)
print(fails, total_games)
