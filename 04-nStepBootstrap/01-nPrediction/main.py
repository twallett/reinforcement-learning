#%%
from Utils.env import GridWorldEnv
from model import nTDPrediction

SEED = 123 
SIZE = 3
GAMMA = 0.9
ALPHA = 0.1
N = 4
EPISODES = 50
FPS = 0

env = GridWorldEnv(render_mode="human",
                   size=SIZE)

env.metadata['render_fps'] = FPS
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n

model = nTDPrediction(num_states = NUM_STATES,
                      num_actions=NUM_ACTIONS,
                      alpha=ALPHA,
                      gamma=GAMMA,
                      seed = SEED)

for episode in range(EPISODES):
    state = env.reset(seed=SEED)
    t = 0
    T = float('inf')
    sampled_states = [state]
    sampled_rewards = []
    done = False
    while not done:
        if t < T:
            action = model.pi()
            next_state, reward, terminated = env.step(action)
            if not terminated:
                sampled_states.append(next_state)
            sampled_rewards.append(reward)
            if terminated:
                T = t + 1
        tau = t - N + 1
        if tau >= 0:
            env.v = model.update(sampled_states, sampled_rewards, tau, N, T, terminated, t)
        env.render()
        state = next_state
        t += 1
        done = terminated
env.close() 
