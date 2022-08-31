import abc


class EpsilonDecayStrategy(abc.ABC):
    @abc.abstractmethod
    def decay_epsilon(self, epsilon: float) -> float:
        return


class LinearEpsilonDecayStrategy(EpsilonDecayStrategy):
    def __init__(self, steps_until_min: int, epsilon_min: float) -> None:
        super().__init__()
        self.steps_until_min = steps_until_min
        self.epsilon_min = epsilon_min

    def decay_epsilon(self, epsilon: float, **kwargs) -> float:
        step = kwargs.get('step')
        subtraction_step = epsilon/self.steps_until_min

        if step:
            return max(epsilon - (step * subtraction_step), self.epsilon_min)
        return max(epsilon-subtraction_step, self.epsilon_min)
