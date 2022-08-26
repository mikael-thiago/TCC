import gym
from gym.wrappers import AtariPreprocessing
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from tensorflow.python.keras.losses import MeanSquaredError

from tensorflow import keras

from agent import DQN, Params
from memory import Memory


def create_network(num_actions):
    network = Sequential([
        Conv2D(32, 8, 4, input_shape=(84, 84, 1), activation='relu'),
        Conv2D(64, 4, 2, activation='relu'),
        Conv2D(64, 3, 1, activation='relu'),
        Flatten(),
        Dense(512),
        Dense(num_actions, activation='linear')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=25e-5, clipvalue=1)

    network.compile(loss=MeanSquaredError(), optimizer=optimizer)
    return network


ENV_NAME = 'SpaceInvadersNoFrameskip-v4'
SAVE_PATH = "models/DQN-"+ENV_NAME

env = gym.make(ENV_NAME, render_mode='human')
env = AtariPreprocessing(env, frame_skip=4, grayscale_newaxis=True)

training_network = create_network(env.action_space.n)
target_network = create_network(env.action_space.n)

agent = DQN(training_network, target_network, Params(num_actions=env.action_space.n,
            epsilon=1, epsilon_min=0.1, epsilon_decay=0.9, update_steps=1000))
replayMemory = Memory(size=400)

try:
    training_network.load_weights(SAVE_PATH)
    agent.updateTargetNetwork()
    print('Models loaded from path {}'.format(SAVE_PATH))
except:
    pass

done = False

state = env.reset()
env.step(1)
steps = 0
acc_reward = 0
lives = 5

while not done:
    action = agent.predict(np.expand_dims(state, axis=0), test=True)
    next_state, reward, dn, _ = env.step(action)
    state = next_state
    done = dn
    steps += 1
    acc_reward += reward

    if(_.get('lives') < lives):
        lives = _.get('lives')
        env.step(1)

    print(reward, acc_reward, _)

print('Reward mean: {:.2f}'.format(acc_reward/steps))
