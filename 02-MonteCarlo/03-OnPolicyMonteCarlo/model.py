#%%
import numpy as np

class MCOnPolicyControl:

    def __init__(self, num_states, num_actions, epsilon = 0.2, gamma = 0.9, seed = 123):
        np.random.seed(seed)
        self.q = np.zeros((num_states, num_actions))
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.returns = {f"{state},{action}": [] for state in range(num_states) for action in range(num_actions)}
    
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
    
    def update(self, sampled_states, sampled_actions, sampled_rewards, t):
        g = 0
        for i in reversed(range(0, t)):
            state = sampled_states[i]
            action = sampled_actions[i]
            reward = sampled_rewards[i]
            g = (self.gamma * g) + reward
            if not any(state == repeat and action == sampled_actions[j] for j, repeat in enumerate(sampled_states[:i])):
                self.returns[f"{state},{action}"].append(g)
                self.q[state, action] = np.mean(self.returns[f"{state},{action}"])
        return self.q
