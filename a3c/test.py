
from agent import A3C
from environment import GymEnvironment
from keras.models import *
from keras.layers import *

ENV_NAME = 'BreakoutNoFrameskip-v4'  # Parametrizar pelo CLI
SAVE_PATH = "models/DQN-"+ENV_NAME

environment = GymEnvironment(env_name=ENV_NAME, render=True)
agent = A3C.from_path(path=SAVE_PATH, env=environment)
agent.play()
