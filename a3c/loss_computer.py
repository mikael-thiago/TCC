from typing import List
import numpy as np

from memory import Memory
import tensorflow as tf

import sys


class LossComputer():
    def __init__(self, entropyBias, v_loss_weight=0.5) -> None:
        self.entropyBias = entropyBias
        self.v_loss_weight = v_loss_weight

    def compute(self, last_state, model, action_size, done: bool, replayMemory: Memory, discount_factor: float):

        discounted_rewards = self.__compute_discounted_rewards(
            rewards=replayMemory.rewards, done=done, last_state=last_state, model=model, discount_factor=discount_factor)

        logits, values = model(
            tf.convert_to_tensor(np.array(replayMemory.states),
                                 dtype=tf.float32))

        d_rewards_tensor = tf.convert_to_tensor(discounted_rewards,
                                                dtype=tf.float32)

        # print('VALUES {}'.format(np.array(values).shape))
        # print('REWARDS {}'.format(np.array(discounted_rewards).shape))

        advantage = d_rewards_tensor - values

        v_loss = advantage**2

        print('V_LOSS {}'.format(np.array(v_loss).shape))

        # policy loss
        policy = tf.nn.softmax(logits)

        actions_one_hot = tf.one_hot(
            replayMemory.actions, action_size, dtype=tf.float32, axis=1)

        policy_bias = 1e-20

        entropy = self.__compute_entropy(policy, policy_bias)

        # print('ACTIONS ONE HOT {}'.format(actions_one_hot))

        p_loss = tf.nn.softmax_cross_entropy_with_logits(labels=actions_one_hot,
                                                         logits=logits)

        p_loss = np.vstack(p_loss) * tf.stop_gradient(advantage)

        # print(p_loss)

        # Subtrai-se entropia do loss, pois o objetivo é maximizá-la para estimular a exploração
        p_loss = p_loss + self.entropyBias * entropy

        # print('ENTROPY {}'.format(entropy))
        # print('SUMMING ENTROPY {}'.format(p_loss + self.entropyBias * entropy))
        # print('SUBTRACTING ENTROPY {}'.format(
        #     p_loss - self.entropyBias * entropy))

        total_loss = tf.reduce_mean((self.v_loss_weight * v_loss) + p_loss)

        return total_loss

    def __compute_discounted_rewards(self, rewards: List, done: bool, model, last_state, discount_factor: float):
        acc = 0

        # Se não for estado terminal,
        # usar rede neural para prever o valor do ultimo estado
        if not done:
            state_tensor = tf.convert_to_tensor(
                np.expand_dims(last_state, axis=0),
                dtype=tf.float32
            )
            _, value = model(state_tensor)
            acc = value.numpy()[0][0]

        discounted_rewards = []

        for reward in rewards[::-1]:
            acc = reward + discount_factor * acc
            discounted_rewards.append(acc)

        discounted_rewards.reverse()

        return np.vstack(discounted_rewards)

    # H(x) = - sum x in P(x) * log(P(x))
    # reference: https://machinelearningmastery.com/cross-entropy-for-machine-learning/
    def __compute_entropy(self, probs, policyBias: float):
        return tf.reduce_sum(probs * tf.math.log(probs + policyBias), axis=1, keepdims=True)
