import numpy as np

class MCOffPolicyControl:

    def __init__(self, num_states, num_actions, epsilon = 0.2, gamma = 0.9, seed = 123):
        np.random.seed(seed)
        self.q = np.zeros((num_states, num_actions))
        self.c = np.zeros((num_states, num_actions))
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma
    
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
    
    def update(self, sampled_states, sampled_actions, sampled_rewards, behavior_probabilities, t):
        g = 0
        w = 1
        for i in reversed(range(0, t)):
            state = sampled_states[i]
            action = sampled_actions[i]
            reward = sampled_rewards[i]
            g = (self.gamma * g) + reward
            self.c[state, action] += w
            self.q[state, action] += (w / self.c[state, action]) * (g - self.q[state, action])
            
            if action != np.argmax(self.q[state, :]):
                break
            
            w *= 1 / behavior_probabilities[i][action]

            if w == 0:
                break
    
        return self.q
    