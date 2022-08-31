from collections import deque
from random import sample


class Memory:
    def __init__(self, size=400, sample_size=32) -> None:
        self.memory = deque(maxlen=size)
        self.size = size
        self.sample_size = sample_size

    def push_experience(self, experience):
        state, action, reward, next_state, done = experience

        self.memory.append([state, action, reward, next_state, done])

    def get_experiences(self):
        return sample(self.memory, self.sample_size)
