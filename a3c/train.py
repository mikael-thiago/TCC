from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv2D, Flatten

import tensorflow as tf
from agent import A3C
from environment import GymEnvironment
from PIL import Image


import numpy as np


def create_a3c_network(num_actions, state_shape, hiddenLayersUnits=[256]):
    baseLayers = [
        Conv2D(16, (8, 8), strides=(4, 4),
               input_shape=state_shape, activation='relu'),
        Conv2D(32, (4, 4), strides=(2, 2), activation='relu'),
    ]

    baseLayers.append(Flatten())

    for units in hiddenLayersUnits:
        baseLayers.append(Dense(units, activation='relu'))

    commonNetwork = Sequential(baseLayers)
    # commonNetwork.build((None, *state_shape))

    actor = Dense(num_actions, activation='linear')(commonNetwork.output)
    critic = Dense(1, activation='linear')(commonNetwork.output)

    model = Model(commonNetwork.inputs, [actor, critic])

    print('Model shape {} AQUI'.format(model.output_shape))

    return model


ENV_NAME = 'BreakoutNoFrameskip-v4'  # Parametrizar pelo CLI
SAVE_PATH = "models/DQN-"+ENV_NAME
EPISODES = 1000  # parametrizar pelo CLI
MAX_STEPS_PER_EPISODE = 1000000  # parametrizar pelo CLI
UPDATE_FREQUENCY = 5  # parametrizar pelo CLI
NUMBER_OF_WORKERS = 4  # parametrizar pelo CLI

env = GymEnvironment(env_name=ENV_NAME)


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

try:
    agent = A3C.from_path(path=SAVE_PATH, env=env, optimizer=optimizer)
except:
    model = create_a3c_network(num_actions=env.get_action_shape()[
                               0], state_shape=(84, 84, 4))
    agent = A3C(model=model, env=env, optimizer=optimizer)

agent.train(
    episodes=EPISODES,
    max_steps_per_episode=MAX_STEPS_PER_EPISODE,
    number_of_workers=NUMBER_OF_WORKERS,
    save_path=SAVE_PATH,
    update_frequency=UPDATE_FREQUENCY,
)
