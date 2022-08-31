import numpy as np
from tensorflow.python.keras.engine.training import Model
from .epsilon_decay_strategy import EpsilonDecayStrategy
from .memory import Memory
from tensorflow.keras.models import Model, load_model, clone_model
from shared.pre_processing import PreProcessing


class DQN():
    def __init__(self, training_network: Model, target_network: Model, env):
        self.training_network = training_network
        self.target_network = target_network
        self.env = env

    def play(self):
        state = self.env.reset()
        state = PreProcessing.replicate_frame(frame=state)
        # state = np.stack((state, state, state, state), axis=2)
        # state = np.reshape([state], (84, 84, 4))

        done = False

        steps = 0
        acc_reward = 0

        while not done:
            action = self.__predict(np.expand_dims(state, axis=0), test=True)
            next_state, reward, done, _ = self.env.step(action)
            next_state = PreProcessing.stack_frame(
                previous_frames=state, new_frame=next_state)
            # next_state = np.append(next_state, state[:, :, :3], axis=2)

            state = next_state

            steps += 1
            acc_reward += reward

            print(reward, acc_reward, _)

        print('Reward mean: {:.2f}'.format(acc_reward/steps))

    def train(
        self,
        steps: int,
        update_steps: int,
        save_steps: int,
        save_path: str,
        epsilon: int,
        epsilon_decay_strat: EpsilonDecayStrategy,
        replay_memory: Memory,
        discount_factor: float = 0.99,
        start_step: int = 0
    ):
        self.__populate_replay_memory(
            replay_memory=replay_memory, epsilon=epsilon)

        print('Starting training')
        print('Training args:\nSteps: {}\nUpdate frequency: {}\nSave frequency: {}\nEpsilon: {}\nDiscount factor: {}\nStart step: {}'.format(
            steps, update_steps, save_steps, epsilon, discount_factor, start_step))

        global_steps = start_step

        episode = 0
        while True:
            state = self.env.reset()
            state = PreProcessing.replicate_frame(state, number_of_frames=4)
            # state = np.stack((state, state, state, state), axis=2)
            # state = np.reshape([state], (84, 84, 4))

            acc_reward = 0
            steps = 0
            step = 0

            done = False

            while not done:
                action = self.__predict(np.expand_dims(
                    state, axis=0), epsilon=epsilon)

                next_state, reward, done, _ = self.env.step(action)
                next_state = PreProcessing.stack_frame(
                    previous_frames=state, new_frame=next_state)
                # next_state = np.append(next_state, state[:, :, :3], axis=2)

                reward = np.clip(reward, -1, 1)

                replay_memory.push_experience(
                    [state, action, reward, next_state, done])

                self.__train_step(
                    experiences=replay_memory,
                    discount_factor=discount_factor
                )

                epsilon = epsilon_decay_strat.decay_epsilon(epsilon)

                self.__save_model_if_necessary(
                    step=global_steps, save_steps=save_steps, save_path=save_path)
                self.__update_target_network_if_necessary(
                    step=global_steps, update_steps=update_steps)

                log_frequency = 10

                if global_steps % log_frequency == 0:
                    print(
                        'Episode {} Step {} Global Step {} - Acc reward: {} Epsilon {}'.format(episode, step, global_steps, acc_reward, epsilon))

                state = next_state
                acc_reward += reward
                steps += 1
                step += 1
                global_steps += 1

            self.__update_target_network()
            episode += 1

    def __save_model_if_necessary(self, step: int, save_steps: int, save_path: str):
        if step != 0 and step % save_steps == 0:
            print('Saving models at step {}'.format(step))
            self.training_network.save(save_path)

    def __update_target_network_if_necessary(self, step: int, update_steps: int):
        if step != 0 and step % update_steps == 0:
            print('Updating target network at step {}'.format(step))
            self.__update_target_network()

    def __populate_replay_memory(self, replay_memory: Memory, epsilon: float):
        state = self.env.reset()

        state = np.stack((state, state, state, state), axis=2)
        state = np.reshape([state], (84, 84, 4))

        for i in range(replay_memory.sample_size):
            action = self.__predict(np.expand_dims(
                state, axis=0), epsilon=epsilon)

            next_state, reward, done, _ = self.env.step(action)
            next_state = np.append(next_state, state[:, :, :3], axis=2)

            replay_memory.push_experience(
                [state, action, reward, next_state, done])
            state = next_state

    def __train_step(self, experiences: Memory, discount_factor: float):
        batch = experiences.get_experiences()

        states = np.array([e[0] for e in batch], dtype='float32')
        actions = np.array([e[1] for e in batch], dtype='int')
        rewards = np.array([e[2] for e in batch], dtype='float32')
        next_states = np.array([e[3] for e in batch], dtype='float32')
        dones = np.array([0 if e[4] else 1 for e in batch], dtype='float32')

        batch_size = len(batch)
        # [0,1,...,batch_size-1]
        batch_indexes = np.arange(batch_size)

        # Q(s)
        q_values = self.training_network.predict(states)

        # Q(s')
        q_values_next = self.training_network.predict(next_states)
        # a'
        actions_next = np.argmax(q_values_next, axis=1)
        # Q'(s')
        q_values_next_target = self.target_network.predict(next_states)
        # Q'(s', a')
        q_values_next_actions = q_values_next_target[batch_indexes, actions_next]

        q_values_updated = q_values

        # discounted_future_reward = { γ * Q'(s', a') if not done
        #                            { 0 if done
        # Q(s, a) = R + discounted_future_reward
        # Como descrito em: https://arxiv.org/pdf/1509.06461.pdf
        q_values_updated[batch_indexes, actions] = rewards[batch_indexes] + \
            (discount_factor *
             q_values_next_actions * dones)

        # Executa o treinamento utilizando um batch de estados e q values atualizados
        self.training_network.fit(
            states,
            np.array(q_values_updated),
            batch_size=batch_size,
            verbose=0
        )

    def __predict(self, inputs, epsilon: float = 0, test: bool = False):
        # flag test indica se o algoritmo está rodando pra valer, nesse caso, não é considerada a ação aleatória
        if (not test) and np.random.rand() < epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = np.argmax(self.training_network.predict(inputs)[0])

        return action

    def __update_target_network(self):
        self.target_network.set_weights(self.training_network.get_weights())

    @staticmethod
    def from_path(path: str, compile_network, env):
        try:
            network = load_model(path)
            compile_network(network)
            target_network = clone_model(network)
            compile_network(network)
            return DQN(training_network=network, target_network=target_network, env=env)
        except:
            raise Exception('Error loading model from path {}'.format(path))
