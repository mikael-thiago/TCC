
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from tensorflow.keras import losses
import tensorflow as tf


def compile_network(network: Model):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=10e-4, epsilon=1e-6, clipnorm=10)
    network.compile(loss=losses.MeanSquaredError(), optimizer=optimizer)


def create_network(num_actions: int, dense_units=[512, 512]):
    network = Sequential([
        Conv2D(16, (8, 8), strides=(4, 4), input_shape=(
            84, 84, 4), activation='relu'),
        Conv2D(32, (4, 4), strides=(2, 2), activation='relu'),
        Flatten(),
        *[Dense(units, activation='relu') for units in dense_units],
        Dense(num_actions, activation='linear')
    ])

    compile_network(network=network)
    return network


model_dict = {
    1: lambda num_actions: create_network(num_actions=num_actions, dense_units=[int(1024/2) for _ in range(2)]),
    2: lambda num_actions: create_network(num_actions=num_actions, dense_units=[int(1024/4) for _ in range(4)]),
    3: lambda num_actions: create_network(num_actions=num_actions, dense_units=[int(1024/8) for _ in range(8)]),
    4: lambda num_actions: create_network(num_actions=num_actions, dense_units=[int(1024/16) for _ in range(16)]),
}
