from queue import Queue
import threading

from loss_computer import LossComputer
from memory import Memory
from gym import Env

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model


class Worker(threading.Thread):
    global_episode = 0
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self, model: Model, result_queue: Queue, global_model: Model, worker_id, env: Env, opt, loss_computer: LossComputer, discount_factor: float,  max_episodes: int, max_steps_per_episode: int, update_freq: int, save_dir: str):
        super(Worker, self).__init__()

        self.model = model
        self.result_queue = result_queue
        self.global_model = global_model
        self.worker_id = worker_id
        self.env = env
        self.ep_loss = 0
        self.opt = opt
        self.loss_computer = loss_computer
        self.discount_factor = discount_factor
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_freq = update_freq
        self.save_dir = save_dir

    def run(self):
        local_episode = 0

        while Worker.global_episode < self.max_episodes:
            replayMemory = Memory()
            state = self.env.reset()

            episode_reward = 0

            local_episode += 1

            self.ep_loss = 0

            state = np.stack((state, state, state, state), axis=2)
            state = np.reshape([state], (84, 84, 4))

            for step in range(self.max_steps_per_episode):
                logits, v = self.model.predict(
                    np.expand_dims(state, axis=0))

                policy = tf.nn.softmax(logits)

                action = np.random.choice(
                    self.env.action_space.n, p=policy.numpy()[0])

                new_state, reward, done, _ = self.env.step(action)
                new_state = np.append(new_state, state[:, :, :3], axis=2)

                episode_reward += reward

                # if Worker.global_episode % 10 == 0 and step == 0:
                print('Worker {}, Action {}, Global Episode {}, Episode {} - Step {}, Reward: {} Lives: {}'.format(
                    self.worker_id, action, Worker.global_episode, local_episode, step + 1, episode_reward, _['lives']))

                replayMemory.store(state, action, reward)

                if ((step+1) % self.update_freq == 0) or done:
                    self.calculate_gradients_and_update_global_model(
                        last_state=new_state, done=done, replayMemory=replayMemory)

                    self.update_global_best_score(
                        done=done, episode_reward=episode_reward)

                if done:
                    break

                state = new_state

            with Worker.save_lock:
                print('Worker {} saving weights'.format(
                    self.worker_id))
                self.global_model.save(self.save_dir)
                Worker.global_episode += 1

            self.result_queue.put(None)

    def calculate_gradients_and_update_global_model(self, last_state, done: bool, replayMemory: Memory):
        with tf.GradientTape() as tape:
            total_loss = self.loss_computer.compute(
                last_state=last_state,
                model=self.model,
                action_size=self.env.action_space.n,
                done=done,
                replayMemory=replayMemory,
                discount_factor=self.discount_factor
            )

            gradients = tape.gradient(
                total_loss, self.model.trainable_weights)

            # print('Gradients {}'.format(np.array(gradients)[0]))

            gradients, _ = tf.clip_by_global_norm(gradients, 40)

            # print('Gradients {}'.format(np.array(gradients)[0]))

            with Worker.save_lock:
                self.opt.apply_gradients(
                    zip(gradients, self.global_model.trainable_weights))
                self.model.set_weights(
                    self.global_model.get_weights())

    def update_global_best_score(self, done: bool, episode_reward):
        if done and episode_reward > Worker.best_score:
            with Worker.save_lock:
                Worker.best_score = episode_reward
