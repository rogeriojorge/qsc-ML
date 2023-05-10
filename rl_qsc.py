import os
import sys
from pathlib import Path
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

params = {'results_path': 'results',
          'algorithm': 'PPO',
          'total_timesteps': 200000,
          'eval_episodes': 100
          }

# Set the environment
env_name = 'CartPole-v1'
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])

# Set the algorithm
if params['algorithm'] == 'PPO':
    model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=params['total_timesteps'])

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=params['eval_episodes'])
print(f'Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')

# Save the model
this_path = str(Path(__file__).parent.resolve())
general_results_path = os.path.join(this_path, params['results_path'])
os.makedirs(general_results_path, exist_ok=True)
model.save(os.path.join(general_results_path, f"{env_name}_{params['algorithm']}.zip"))

# Load the saved model and test it
loaded_model = PPO.load(os.path.join(general_results_path, f"{env_name}_{params['algorithm']}.zip"))

obs = env.reset()
for _ in range(1000):
    action, _states = loaded_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()