import abc
from typing import Any, Tuple

import gym
from gym.wrappers import AtariPreprocessing


class Environment(abc.ABC):
    @abc.abstractmethod
    def get_observation_shape(self) -> Tuple:
        return

    @abc.abstractmethod
    def get_action_shape(self) -> Tuple:
        return

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool]:
        return

    @abc.abstractmethod
    def reset(self) -> Any:
        return

    @abc.abstractmethod
    def clone(self) -> Any:
        return

    @abc.abstractmethod
    def render(self, **kwargs) -> None:
        return

    @abc.abstractmethod
    def close(self) -> None:
        return


class GymEnvironment(Environment):
    def __init__(self, env_name: str, render: bool = False) -> None:
        super().__init__()
        self.env_name = env_name
        self.env = self.__create_env(env_name, render)

    def __create_env(self, env_name: str, render: bool = False):
        env = gym.make(env_name)

        if render:
            env = gym.make(env_name, render_mode='human')

        return AtariPreprocessing(
            env, frame_skip=1, grayscale_obs=True, grayscale_newaxis=True)

    def get_observation_shape(self) -> Tuple:
        return self.env.observation_space.shape

    def get_action_shape(self) -> Tuple:
        return tuple([self.env.action_space.n])

    def step(self, action: int) -> Tuple[Any, float, bool]:
        return self.env.step(action)

    def reset(self) -> Any:
        return self.env.reset()

    def clone(self) -> Any:
        return self.__create_env(env_name=self.env_name)

    def render(self, **kwargs) -> None:
        self.env.render(kwargs)

    def close(self) -> None:
        self.env.close()
