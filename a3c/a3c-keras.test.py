import gym
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential, load_model, clone_model

model = load_model('cart-pole-keras')
model.compile()

env = gym.make("CartPole-v0")  # Create the environment

env.render(mode='human')
state = env.reset()

while True:
  state = tf.convert_to_tensor(state)
  state = tf.expand_dims(state, 0)

  policy, _ = model(state)
  action = np.argmax(policy)

  state, reward, done, info = env.step(action)
  