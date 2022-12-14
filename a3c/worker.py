from queue import Queue
import threading
from typing import Any, Tuple

from .loss_computer import LossComputer
from .memory import Memory
from gym import Env

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from shared.pre_processing import PreProcessing


class Worker(threading.Thread):
    global_step = 0
    global_moving_average_reward = 0
    best_score = 0

    def __init__(self,
                 model: Model,
                 result_queue: Queue,
                 global_model: Model,
                 worker_id,
                 env: Env,
                 opt,
                 loss_computer: LossComputer,
                 discount_factor: float,
                 update_freq: int,
                 save_dir: str,
                 max_steps: int,
                 save_lock: threading.Lock):
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
        self.update_freq = update_freq
        self.save_dir = save_dir
        self.max_steps = max_steps
        self.save_lock = save_lock

    def run(self):
        max_global_steps = self.max_steps

        episode_reward = 0
        episode = 1

        state = self.__reset()

        while Worker.global_step < max_global_steps:
            steps_performed, done, memory, last_state, acc_reward = self.__perform_steps(
                self.update_freq, state)

            episode_reward += acc_reward

            gradients = self.__calculate_gradients(
                last_state=last_state, done=done, replay_memory=memory)

            with self.save_lock:
                self.__update_global_model(gradients)
                print('Worker {} Episode {} Reward {} Global step {}'.format(
                    self.worker_id, episode, episode_reward, Worker.global_step))
                self.global_model.save(self.save_dir)

                Worker.global_step += steps_performed

            if done:
                state = self.__reset()
                episode_reward = 0
                episode += 1
            else:
                state = last_state

    def __perform_steps(self, steps: int, initial_state) -> Tuple[int, bool, Any, Any, float]:
        replay_memory = Memory()
        state = initial_state
        steps_performed = 0
        acc_reward = 0
        done = False

        for step in range(steps):
            logits, _ = self.model.predict(
                np.expand_dims(state, axis=0))

            policy = tf.nn.softmax(logits)

            action = np.random.choice(
                self.env.action_space.n, p=policy.numpy()[0])

            # Concatena o novo estado aos ultimos 4 frames e remove o primeiro
            new_state, reward, done, _ = self.env.step(action)
            new_state = PreProcessing.stack_frame(
                previous_frames=state, new_frame=new_state)
            # new_state = np.append(new_state, state[:, :, :3], axis=2)

            # Coloca recompensa entre 1 e -1
            reward = np.clip(reward, -1, 1)

            replay_memory.store(state, action, reward)

            steps_performed += 1
            acc_reward += reward

            if done:
                break

            state = new_state

        return steps_performed, done, replay_memory, state, acc_reward

    def __calculate_gradients(self, last_state, done: bool, replay_memory: Memory):
        with tf.GradientTape(persistent=True) as tape:
            # Computa loss
            total_loss = self.loss_computer.compute(
                last_state=last_state,
                model=self.model,
                action_size=self.env.action_space.n,
                done=done,
                replay_memory=replay_memory,
                discount_factor=self.discount_factor
            )

            # Computa gradientes
            gradients = tape.gradient(
                total_loss, self.model.trainable_variables)

            # Limita os gradientes para evitar degenera????o do algoritmo
            # gradients, _ = tf.clip_by_global_norm(gradients, 40)

            return gradients

    def __update_global_model(self, gradients):
        # Aplica gradients a rede global e
        # atualiza os pesos da rede local
        self.opt.apply_gradients(
            zip(gradients, self.global_model.trainable_variables))
        self.model.set_weights(
            self.global_model.get_weights())

    def __update_global_best_score(self, done: bool, episode_reward):
        if done and episode_reward > Worker.best_score:
            with self.save_lock:
                Worker.best_score = episode_reward

    def __reset(self):
        state = self.env.reset()
        # Ser?? passado para a rede neural os ??ltimos 4 frames coletados
        # Assim ?? poss??vel ela considerar informa????es derivadas como, por exemplo:
        # Dire????o da bola no jogo Breakout, dire????o dos aliens no Space Invaders
        # Para o primeiro passo, como n??o h?? frames anteriores, ?? utilizado o primeiro 4 vezes
        return PreProcessing.replicate_frame(state, number_of_frames=4)
        # state = np.stack((state, state, state, state), axis=2)
        # state = np.reshape([state], (84, 84, 4))
        # return state
