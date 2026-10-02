#%%
from Utils.env import GridWorldEnv
from model import TDSARSA

SEED = 123 
SIZE = 3
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.2
EPISODES = 100
FPS = 0

env = GridWorldEnv(render_mode="human",
                   size=SIZE)

env.metadata['render_fps'] = FPS
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n

model = TDSARSA(num_states = NUM_STATES,
                num_actions=NUM_ACTIONS,
                epsilon=EPSILON,
                alpha=ALPHA,
                gamma=GAMMA,
                seed = SEED)

for episode in range(EPISODES):
    state = env.reset(seed=SEED)
    action = model.esoft(state, NUM_ACTIONS)
    done = False
    while not done:
        next_state, reward, terminated = env.step(action)
        next_action = model.esoft(next_state, NUM_ACTIONS)
        env.q = model.update(action, next_action, state, next_state, reward)
        env.render()
        action = next_action
        state = next_state
        done = terminated
env.close() 