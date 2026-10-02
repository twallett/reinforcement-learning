#%%
from Utils.env import GridWorldEnv
from model import nSARSA

SEED = 123 
SIZE = 3
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.2
N = 4
EPISODES = 50
FPS = 0

env = GridWorldEnv(render_mode="human",
                   size=SIZE)

env.metadata['render_fps'] = FPS
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n

model = nSARSA(num_states = NUM_STATES,
               num_actions=NUM_ACTIONS,
               alpha=ALPHA,
               epsilon = EPSILON,
               gamma=GAMMA,
               seed = SEED)

for episode in range(EPISODES):
    state = env.reset(seed=SEED)
    action = model.esoft(state, NUM_ACTIONS)
    t = 0
    T = float('inf')
    sampled_states = [state]
    sampled_actions = [action]
    sampled_rewards = []
    done = False
    while not done:
        if t < T:
            next_state, reward, terminated = env.step(action)
            if not terminated:
                sampled_states.append(next_state)
            sampled_rewards.append(reward)
            if terminated:
                T = t + 1
            else:
                next_action = model.esoft(next_state, NUM_ACTIONS)
                sampled_actions.append(next_action)
        tau = t - N + 1
        if tau >= 0:
            env.q = model.update(sampled_states, sampled_actions, sampled_rewards, tau, N, T, terminated, t)
        env.render()
        action = next_action
        state = next_state
        t += 1
        done = terminated
env.close() 