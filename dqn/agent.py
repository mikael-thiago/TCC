import numpy as np
from tensorflow.python.keras.engine.training import Model
from memory import Memory


class Params():
    def __init__(self, num_actions, epsilon, epsilon_min, epsilon_decay, update_steps, gamma=0.99, lr=1e-3):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.num_actions = num_actions
        self.update_steps = update_steps


class DQN():
    def __init__(self, training_network: Model, target_network: Model, params: Params):
        self.params = params
        self.training_network = training_network
        self.target_network = target_network

    def train(self, experiences: Memory):
        target_q_values = []

        # batchs de experiencia
        for experience in experiences.get_experiences():
            state, action, reward, next_state, done = experience

            q_values_curr = self.training_network.predict(
                np.expand_dims(state, axis=0))[0]

            best_action_next = self.predict(np.expand_dims(next_state, axis=0))

            q_values_next_target = self.target_network.predict(
                np.expand_dims(next_state, axis=0))[0]

            # aqui está meio anti intuitivo, done é 0 quando é um estado terminal.
            # isso foi feito por questão de conveniencia, pois o segundo termo da soma é zerado quando o estado é terminal,
            # portanto, o armazenei como 0.
            q_value_update = reward + (self.params.gamma *
                                       q_values_next_target[best_action_next] * done)

            q_values_curr[action] = q_value_update

            # armazeno os q_values resultados em uma lista
            target_q_values.append(q_values_curr)

        # armazeno os estados do ambiente em outra lista
        states = np.array([e[0] for e in experiences.get_experiences()])

        self.decayEpsilon()

        # executa o gradiente utilizando um array de estados e um array de q values
        # states = [estado], target_q_values = [q_value]
        # é feito o treinamento associando o states[i] à target_q_values[i]
        self.training_network.fit(states, np.array(target_q_values), verbose=0)

    def predict(self, inputs, test=False):
        # flag test indica se o algoritmo está rodando pra valer, nesse caso, não é considerada a ação aleatória
        if (not test) and np.random.rand() < self.params.epsilon:
            action = np.random.randint(0, self.params.num_actions)
        else:
            action = np.argmax(self.training_network.predict(inputs)[0])

        return action

    def decayEpsilon(self):
        self.params.epsilon = max(
            self.params.epsilon*self.params.epsilon_decay, self.params.epsilon_min)

    def updateTargetNetwork(self):
        self.target_network.set_weights(self.training_network.get_weights())
