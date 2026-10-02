#%%
import numpy as np

class ValueIteration:
    
    def __init__(self, num_states, gamma = 0.9, seed = 123):
        np.random.seed(seed)
        self.gamma = gamma
        self.v = np.zeros((num_states))

    def update(self, state, next_states, rewards):
        v = []
        for next_state, reward in zip(next_states, rewards):
            v.append( 1 * (reward + (self.gamma * self.v[next_state])) )
        self.v[state] = np.max(v)
        return self.v