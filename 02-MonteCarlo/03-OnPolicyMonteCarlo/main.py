#%%
from Utils.env import GridWorldEnv
from model import MCOnPolicyControl

SEED = 123 
SIZE = 3
GAMMA = 0.9
EPSILON = 0.2
EPISODES = 100
FPS = 0

env = GridWorldEnv(render_mode="human",
                   size=SIZE)

env.metadata['render_fps'] = FPS
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n

model = MCOnPolicyControl(num_states = NUM_STATES,
                          num_actions=NUM_ACTIONS,
                          epsilon=EPSILON, 
                          gamma=GAMMA,
                          seed = SEED)

for episode in range(EPISODES):
    state = env.reset(seed=SEED)
    done = False
    sampled_states = [state]
    sampled_actions = []
    sampled_rewards = []
    t = 0 
    while not done:
        action = model.esoft(state, NUM_ACTIONS)
        state, reward, terminated = env.step(action)
        if not terminated:
            sampled_states.append(state)
        sampled_actions.append(action)
        sampled_rewards.append(reward)
        t += 1
        env.render()
        done = terminated
    env.q = model.update(sampled_states, sampled_actions, sampled_rewards, t)
env.close() 