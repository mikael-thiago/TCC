import gym
from gym.wrappers import AtariPreprocessing

from .agent import DQN
from .networks import compile_network


ENV_NAME = 'BreakoutNoFrameskip-v4'
SAVE_PATH = "models/DQN-"+ENV_NAME

env = gym.make(ENV_NAME, render_mode='human')
env = AtariPreprocessing(env, frame_skip=4, grayscale_newaxis=True)

agent = DQN.from_path(env=env, compile_network=compile_network, path=SAVE_PATH)

agent.play()
