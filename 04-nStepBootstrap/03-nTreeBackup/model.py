#%%
import numpy as np

class nTree:

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
    
    def update(self, target, sampled_states, sampled_actions, sampled_rewards, tau, n, T, terminated, t):
        
        if t + 1 >= T:
            g = sampled_rewards[-1]
        else:
            g = sampled_rewards[-1] + (self.gamma * sum([target[-1][i] * self.q[sampled_states[-1], i] for i in range(4)])) 
            for k in reversed(range(tau+1, min(t, T))):
                    g = sampled_rewards[k] + (self.gamma * sum([target[k][i] * self.q[sampled_states[k],i] for i in range(4) if i != sampled_actions[k]])) + (self.gamma * target[k][sampled_actions[k]] * g)
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
                    if t + 1 >= T:
                        g = sampled_rewards[-1]
                    else:
                        g = sampled_rewards[-1] + (self.gamma * sum([target[-1][i] * self.q[sampled_states[-1], i] for i in range(4)])) 
                        for k in reversed(range(tau+1, min(t, T))):
                                g = sampled_rewards[k] + (self.gamma * sum([target[k][i] * self.q[sampled_states[k],i] for i in range(4) if i != sampled_actions[k]])) + (self.gamma * target[k][sampled_actions[k]] * g)
                    state_tau = sampled_states[tau]
                    action_tau = sampled_actions[tau]
                    self.q[state_tau, action_tau] += self.alpha * (g - self.q[state_tau, action_tau])
                
        return self.q
