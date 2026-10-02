#%%
import numpy as np

class TDSARSA:

    def __init__(self, num_states, num_actions, epsilon = 0.2, alpha = 0.1, gamma = 0.9, seed = 123):
        np.random.seed(seed)
        self.q = np.zeros((num_states, num_actions))
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
    
    def esoft(self, state, num_actions, return_probabilities=False):
        
        greedy_action = np.argmax(self.q[state,:])
        
        behavior_probabilities = np.ones(num_actions) * (self.epsilon / num_actions)
        behavior_probabilities[greedy_action] += (1 - self.epsilon)
        
        selected_action = np.random.choice(np.arange(num_actions), p=behavior_probabilities)
        
        target_probabilities = np.zeros(num_actions)
        target_probabilities[greedy_action] = 1
        
        if return_probabilities:
            return selected_action, behavior_probabilities, target_probabilities
        else:
            return selected_action
    
    def update(self, action, next_action, state, next_state, reward):
        self.q[state, action] += self.alpha * (reward + (self.gamma * self.q[next_state, next_action]) - self.q[state, action]) 
        return self.q