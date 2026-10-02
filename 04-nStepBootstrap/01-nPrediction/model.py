#%%
import numpy as np

class nTDPrediction:

    def __init__(self, num_states, num_actions, alpha = 0.1, gamma = 0.9, seed = 123):
        np.random.seed(seed)
        self.v = np.zeros(num_states)
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
    
    def pi(self):
        return np.random.randint(self.num_actions)
    
    def update(self, sampled_states, sampled_rewards, tau, n, T, terminated, t):
        
        g = sum([(self.gamma ** (i-tau-1)) * sampled_rewards[i-1] for i in range(tau+1, min(tau+n, T)+1)])
        if tau + n < T:
            state_tau_n = sampled_states[tau+n]
            g += (self.gamma ** n) * self.v[state_tau_n]
        state_tau = sampled_states[tau]
        self.v[state_tau] += self.alpha * (g - self.v[state_tau])
        
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
                        g += (self.gamma ** n) * self.v[state_tau_n]
                    state_tau = sampled_states[tau]
                    self.v[state_tau] += self.alpha * (g - self.v[state_tau])
                
        return self.v
