class Memory:
    def __init__(self, max_size=80000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.max_size = max_size

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

        if len(self.states) > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
