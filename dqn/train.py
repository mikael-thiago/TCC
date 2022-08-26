import gym
from gym.wrappers import AtariPreprocessing
from collections import deque
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from tensorflow.python.keras.losses import MeanSquaredError

from agent import DQN, Params
from memory import Memory

import tensorflow as tf


def create_network(num_actions):
    network = Sequential([
        Conv2D(32, 8, 4, input_shape=(84, 84, 1), activation='relu'),
        Conv2D(64, 4, 2, activation='relu'),
        Conv2D(64, 3, 1, activation='relu'),
        Flatten(),
        Dense(512),
        Dense(num_actions, activation='linear')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=25e-5, clipvalue=1)

    network.compile(loss=MeanSquaredError(), optimizer=optimizer)
    return network


ENV_NAME = 'SpaceInvadersNoFrameskip-v4'
SAVE_PATH = "models/DQN-"+ENV_NAME

env = gym.make(ENV_NAME)
# Wrapper para transformar a imagem do atari de 210x160 com 128 bits de cor para 84x84 em escala de cinza
# 210x160x128 -> 84x84x1
env = AtariPreprocessing(
    env, frame_skip=4, grayscale_obs=True, grayscale_newaxis=True)

env.seed(123)

training_network = create_network(env.action_space.n)
target_network = create_network(env.action_space.n)

agent = DQN(training_network, target_network, Params(num_actions=env.action_space.n,
            epsilon=1, epsilon_min=0.1, epsilon_decay=0.9, update_steps=500))
replayMemory = Memory(size=400, sample_size=100)

try:
    training_network.load_weights(SAVE_PATH)
    agent.updateTargetNetwork()
    print('Models loaded from path {}'.format(SAVE_PATH))
except:
    pass

print('Populating replay memory')

state = env.reset()

# Popula replay memory utilizando ações aleatórias e armazenando os resutlados
for i in range(replayMemory.size):
    action = agent.predict(np.expand_dims(state, axis=0))

    next_state, reward, done, _ = env.step(action)

    replayMemory.push_experience([state, action, reward, next_state, done])
    state = next_state

episodes = 100
episode_steps = 10000
save_steps = 5

print('Starting training')

for i in range(episodes):
    print('Episode {}'.format(i + 1))

    state = env.reset()
    acc_reward = 0
    steps = 0

    for j in range(episode_steps):
        action = agent.predict(np.expand_dims(state, axis=0))

        next_state, reward, done, _ = env.step(action)

        replayMemory.push_experience([state, action, reward, next_state, done])

        agent.train(replayMemory)

        if j != 0 and j % save_steps == 0:
            print('Saving models at step {}'.format(j))
            training_network.save_weights(SAVE_PATH)

        acc_reward += reward
        steps += 1

        if j != 0 and j % agent.params.update_steps == 0:
            print('Updating target network at step {}'.format(j))
            agent.updateTargetNetwork()

        print('Step {} - Acc reward: {}'.format(j, acc_reward))
        print('Done: {} Lives {}'.format(done, _.get('lives')))

        state = next_state

        if done:
            print('Done, so updating target network')
            agent.updateTargetNetwork()
            break
