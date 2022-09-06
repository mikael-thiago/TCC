import abc


class EpsilonDecayStrategy(abc.ABC):
    @abc.abstractmethod
    def decay_epsilon(self, epsilon: float) -> float:
        return


class LinearEpsilonDecayStrategy(EpsilonDecayStrategy):
    def __init__(self, steps_until_min: int, epsilon_initial: float, epsilon_min: float) -> None:
        super().__init__()
        self.epsilon_min = epsilon_min
        self.subtraction_step = (epsilon_initial-epsilon_min)/steps_until_min

    def decay_epsilon(self, epsilon: float) -> float:
        return max(epsilon-self.subtraction_step, self.epsilon_min)

    def get_epsilon(self, epsilon: float, step: int) -> float:
        return max(epsilon - (step * self.subtraction_step), self.epsilon_min)
