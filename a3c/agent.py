
from queue import Queue
from tensorflow.keras.models import Model, Sequential, load_model, clone_model

from environment import Environment
from loss_computer import LossComputer
from worker import Worker
import tensorflow as tf
import numpy as np


class A3C():
    def __init__(self, model: Model, env: Environment, optimizer=None) -> None:
        self.model = model
        self.env = env
        self.optimizer = optimizer

        # self.model.compile(optimizer=optimizer, loss=[self.loss])

        pass

    def loss(self, *args):
        print(args)

    def train(self, max_steps_per_episode: int, episodes: int, update_frequency: int, save_path: str, number_of_workers: int):
        result_queue = Queue()

        if not self.optimizer:
            raise Exception(
                'The agent cannot be trained, because it has no optimizer')

        workers = [
            Worker(
                model=clone_model(self.model),
                result_queue=result_queue,
                env=self.env.clone(),
                discount_factor=.99,
                global_model=self.model,
                loss_computer=LossComputer(entropyBias=.01),
                opt=self.optimizer,
                worker_id=i,
                max_episodes=episodes,
                max_steps_per_episode=max_steps_per_episode,
                update_freq=update_frequency,
                save_dir=save_path
            ) for i in range(number_of_workers)
        ]

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        return

    def play(self):
        memory = []

        self.env.render()

        state = self.env.reset()

        state = np.stack((state, state, state, state), axis=2)
        state = np.reshape([state], (84, 84, 4))

        memory = state

        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                policy, _ = self.model(tf.convert_to_tensor(
                    memory[None, :], dtype=tf.float32))
                policy = tf.nn.softmax(policy)
                action = np.argmax(policy)

                new_state, reward, done, info = self.env.step(action)

                memory = np.append(new_state, memory[:, :, :3], axis=2)

                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(
                    step_counter, reward_sum, action))
                step_counter += 1

        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            self.env.close()

    @staticmethod
    def from_path(path: str, env: Environment, optimizer=None):
        try:
            model = load_model(path)
            model.compile()
            print('Model shape {}'.format(model.input_shape))
            return A3C(model=model, env=env, optimizer=optimizer)
        except:
            raise Exception('Error loading model from path {}'.format(path))
