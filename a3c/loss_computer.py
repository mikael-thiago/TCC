from typing import List
import numpy as np

from memory import Memory
import tensorflow as tf

import sys


class LossComputer():
    def __init__(self, entropyBias, v_loss_weight=0.5) -> None:
        self.entropyBias = entropyBias
        self.v_loss_weight = v_loss_weight

    def compute(self, last_state, model, action_size: int, done: bool, replay_memory: Memory, discount_factor: float):

        discounted_rewards = self.__compute_discounted_rewards(
            rewards=replay_memory.rewards, done=done, last_state=last_state, model=model, discount_factor=discount_factor)

        # Logits Shape (None, action_space_size)
        # Values Shape (None, )
        logits, values = model(
            tf.convert_to_tensor(np.array(replay_memory.states),
                                 dtype=tf.float32))

        d_rewards_tensor = tf.convert_to_tensor(discounted_rewards,
                                                dtype=tf.float32)

        # Shape (None, )
        advantage = d_rewards_tensor - values

        # Shape (None, )
        v_loss = advantage**2

        # Shape (None, action_space_size)

        # POLICY LOSS
        policy = tf.nn.softmax(logits)

        policy_bias = 1e-20

        # Shape (None, )
        entropy = self.__compute_entropy(policy, policy_bias)

        # Shape (action_size, action_space_size)
        # Lista de listas com 0 em todas as posições, com exceção da posição ação selecionada
        actions_one_hot = tf.one_hot(
            replay_memory.actions, action_size, dtype=tf.float32, axis=1)

        # Função do tensorflow que calcula:
        # -sum ( log(policy + bias) * actions_one_hot )
        # sendo que, policy = softmax(logits)
        p_loss = tf.nn.softmax_cross_entropy_with_logits(labels=actions_one_hot,
                                                         logits=logits)

        # Shape (None, )
        p_loss = p_loss * tf.stop_gradient(advantage)

        # Subtrai-se entropia do loss, pois o objetivo é maximizá-la para estimular a exploração
        p_loss = p_loss - self.entropyBias * entropy

        # print('V_LOSS {}'.format(v_loss))
        # print('P_LOSS {}'.format(p_loss))
        # print('COMBINED_LOSS {}'.format((self.v_loss_weight * v_loss) + p_loss))
        print('REDUCED_SUM_LOSS {}'.format(tf.reduce_sum((self.v_loss_weight * v_loss) + p_loss)))
        # print('REDUCED_MEAN_LOSS {}'.format(tf.reduce_mean((self.v_loss_weight * v_loss) + p_loss)))

        total_loss = tf.reduce_sum((self.v_loss_weight * v_loss) + p_loss)

        # print('LOSS REDUCED {}'.format(total_loss))

        return total_loss

    def __compute_discounted_rewards(self, rewards: List, done: bool, model, last_state, discount_factor: float):
        acc = 0

        # Se não for estado terminal,
        # usar rede neural para prever o valor do 'future discounted reward' a partir do ultimo estado
        if not done:
            state_tensor = tf.convert_to_tensor(
                np.expand_dims(last_state, axis=0),
                dtype=tf.float32
            )
            _, value = model(state_tensor)
            acc = value.numpy()[0]

        discounted_rewards = []

        for reward in rewards[::-1]:
            acc = reward + discount_factor * acc
            discounted_rewards.append(acc)

        discounted_rewards.reverse()

        # Shape (None, )
        return discounted_rewards

    # H(x) = - sum x in P(x) * log(P(x))
    # reference: https://machinelearningmastery.com/cross-entropy-for-machine-learning/
    def __compute_entropy(self, probs, policyBias: float):
        return -tf.reduce_sum(probs * tf.math.log(probs + policyBias), axis=1)
