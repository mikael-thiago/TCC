import argparse
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Reshape, Dense, Conv2D, Flatten

import tensorflow as tf
from .agent import A3C
from .environment import GymEnvironment


def create_a3c_network(num_actions, state_shape, hiddenLayersUnits=[256]):
    # Convolutional layers como descrito em mnih 2015
    # http://proceedings.mlr.press/v48/mniha16-supp.pdf
    baseLayers = [
        Conv2D(16, (8, 8), strides=(4, 4),
               input_shape=state_shape, activation='relu'),
        Conv2D(32, (4, 4), strides=(2, 2), activation='relu'),
    ]

    baseLayers.append(Flatten())

    for units in hiddenLayersUnits:
        baseLayers.append(Dense(units, activation='relu'))

    commonNetwork = Sequential(baseLayers)

    actor = Dense(num_actions, activation='linear')(commonNetwork.output)
    critic = Dense(1, activation='linear')(commonNetwork.output)
    # Modificando formato de retorno de [[float]] para [float]
    critic = Reshape(())(critic)

    model = Model(commonNetwork.inputs, [actor, critic])

    print('Model output shape {}'.format(model.output_shape))

    return model


parser = argparse.ArgumentParser(prog="TCC-Mikael-A3C")

parser.add_argument('--env', type=str)
parser.add_argument('--model', default=1, type=int)
parser.add_argument('--steps', default=int(1e6), type=int)
parser.add_argument('--update_frequency', default=5, type=int)
parser.add_argument('--number_of_workers', default=4, type=int)

args = parser.parse_args()

ENV_NAME = args.env  # Parametrizar pelo CLI
SAVE_PATH = 'models/{}/A3C-{}'.format(args.model, ENV_NAME)
MAX_STEPS = args.steps
UPDATE_FREQUENCY = args.update_frequency  # parametrizar pelo CLI
NUMBER_OF_WORKERS = args.number_of_workers  # parametrizar pelo CLI

env = GymEnvironment(env_name=ENV_NAME)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

try:
    agent = A3C.from_path(path=SAVE_PATH, env=env)
except:
    model = create_a3c_network(num_actions=env.get_action_shape()[
                               0], state_shape=(84, 84, 4))
    agent = A3C(model=model, env=env)

agent.train(
    max_steps=MAX_STEPS,
    number_of_workers=NUMBER_OF_WORKERS,
    save_path=SAVE_PATH,
    update_frequency=UPDATE_FREQUENCY,
    optimizer=optimizer
)
