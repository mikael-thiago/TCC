import gym
from gym.wrappers import AtariPreprocessing

from shared.gpu import GpuUtils

from .agent import DQN
from .epsilon_decay_strategy import LinearEpsilonDecayStrategy
from .memory import Memory

from .networks import model_dict

import argparse

parser = argparse.ArgumentParser(prog="TCC-Mikael-DDQN")
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--model', default=1, type=int)
parser.add_argument('--steps', default=int(1e6), type=int)
parser.add_argument('--epsilon', default=1, type=int)
parser.add_argument('--epsilon_min', default=0.05, type=float)
parser.add_argument('--discount_factor', default=0.99, type=float)
parser.add_argument('--update_frequency', default=1000, type=int)
parser.add_argument('--save_frequency', default=10000, type=int)
parser.add_argument('--start_step', default=0, type=int)
parser.add_argument('--memory_size', default=int(1e5), type=int)
parser.add_argument('--memory_batch_size', default=32, type=int)
# 5GB of max GPU memory for default
parser.add_argument('--max_memory', default=1024*5, type=int)

args = parser.parse_args()


ENV_NAME = args.env
SAVE_DIR = 'dqn/models/{}/DQN-{}'.format(args.model, ENV_NAME)
LOG_DIR = 'logs/dqn/{}/{}'.format(args.model, ENV_NAME)
STEPS = args.steps
UPDATE_FREQUENCY = args.update_frequency
SAVE_FREQUENCY = args.save_frequency
DISCOUNT_FACTOR = args.discount_factor
START_STEP = args.start_step
MEMORY_SIZE = args.memory_size
MEMORY_BATCH_SIZE = args.memory_batch_size
EPSILON = args.epsilon
EPSILON_MIN = args.epsilon_min
STEPS_UNTIL_EPSILON_MIN = int(STEPS*0.3)
MAX_MEMORY = args.max_memory

if MAX_MEMORY:
    GpuUtils.limit_memory_usage(max_memory=MAX_MEMORY)

env = gym.make(ENV_NAME)
env = AtariPreprocessing(
    env, frame_skip=4, grayscale_obs=True, grayscale_newaxis=True)

env.seed(123)

# try:
#     agent = DQN.from_path(
#         env=env, compile_network=compile_network, path=SAVE_DIR)
# except:
#     print('Couldn\'t load the model, creating one!')
training_network = model_dict[args.model](env.action_space.n)
target_network = model_dict[args.model](env.action_space.n)

target_network.set_weights(training_network.get_weights())

agent = DQN(
    training_network,
    target_network,
    env=env
)

replay_memory = Memory(size=MEMORY_SIZE, sample_size=MEMORY_BATCH_SIZE)

linear_epsilon_decay_strat = LinearEpsilonDecayStrategy(
    steps_until_min=STEPS_UNTIL_EPSILON_MIN, epsilon_initial=EPSILON, epsilon_min=EPSILON_MIN)

epsilon = linear_epsilon_decay_strat.get_epsilon(
    epsilon=EPSILON, step=START_STEP)

agent.train(
    steps=STEPS,
    update_steps=UPDATE_FREQUENCY,
    save_steps=SAVE_FREQUENCY,
    save_path=SAVE_DIR,
    epsilon=epsilon,
    epsilon_decay_strat=linear_epsilon_decay_strat,
    discount_factor=DISCOUNT_FACTOR,
    start_step=START_STEP,
    replay_memory=replay_memory,
    log_dir=LOG_DIR
)
