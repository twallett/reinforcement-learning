#%%
import numpy as np

class nSARSA:

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
    
    def update(self, sampled_states, sampled_actions, sampled_rewards, tau, n, T, terminated, t):
        
        g = sum([(self.gamma ** (i-tau-1)) * sampled_rewards[i-1] for i in range(tau+1, min(tau+n, T)+1)])
        if tau + n < T:
            state_tau_n = sampled_states[tau+n]
            action_tau_n = sampled_actions[tau+n]
            g += (self.gamma ** n) * self.q[state_tau_n, action_tau_n]
        state_tau = sampled_states[tau]
        action_tau = sampled_actions[tau]
        self.q[state_tau, action_tau] += self.alpha * (g - self.q[state_tau, action_tau])
        
        if terminated:
            while True:
                t += 1
                tau = t - n + 1
                if tau == T:
                    break
                if tau >= 0:
                    g = sum([(self.gamma ** (i-tau-1)) * sampled_rewards[i-1] for i in range(tau+1, min(tau+n, T)+1)])
                    if tau + n < T:
                        state_tau_n = sampled_states[tau+n]
                        action_tau_n = sampled_actions[tau+n]
                        g += (self.gamma ** n) * self.q[state_tau_n, action_tau_n]
                    state_tau = sampled_states[tau]
                    action_tau = sampled_actions[tau]
                    self.q[state_tau, action_tau] += self.alpha * (g - self.q[state_tau, action_tau])
                
        return self.q
