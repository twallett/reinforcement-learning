#%%
import numpy as np

class TDDoubleQ:

    def __init__(self, num_states, num_actions, epsilon = 0.2, alpha = 0.1, gamma = 0.9, seed = 123):
        np.random.seed(seed)
        self.q1 = np.zeros((num_states, num_actions))
        self.q2 = np.zeros((num_states, num_actions))
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
    
    def esoft(self, state, num_actions, return_probabilities=False):
        
        greedy_action = np.argmax(self.q1[state,:] + self.q2[state,:])
        
        behavior_probabilities = np.ones(num_actions) * (self.epsilon / num_actions)
        behavior_probabilities[greedy_action] += (1 - self.epsilon)
        
        selected_action = np.random.choice(np.arange(num_actions), p=behavior_probabilities)
        
        target_probabilities = np.zeros(num_actions)
        target_probabilities[greedy_action] = 1
        
        if return_probabilities:
            return selected_action, behavior_probabilities, target_probabilities
        else:
            return selected_action
    
    def update(self, action, state, next_state, reward):
        if np.random.randn() > 0.5:
            self.q1[state, action] += self.alpha * (reward + (self.gamma * self.q2[next_state, np.argmax(self.q1[next_state, :])] - self.q1[state, action]))
        else:
            self.q2[state, action] += self.alpha * (reward + (self.gamma * self.q1[next_state, np.argmax(self.q2[next_state, :])] - self.q2[state, action]))
        return self.q1 + self.q2