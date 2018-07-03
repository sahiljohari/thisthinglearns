import gym
import numpy as np
import time
from IPython.display import display, clear_output

n_states = 40

def obs_to_state(env, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    # env.seed(0)
    # np.random.seed(0)

    Q = np.zeros((n_states, n_states, 2))
    G = 0 # accumulated reward
    eta = 0.8 #learning rate

    for episode in range(1000):
        time.sleep(0.1)
        done = False
        state = env.reset()
        reward = 0

        while not done:
            a, b = obs_to_state(env, state)
            action = np.argmax(a, b)
            state_, reward, done, info = env.step(action)

            a_, b_ = obs_to_state(env, state_)
            Q[a][b][action] += eta * (reward + np.max(Q[a_][b_]) - Q[a][b][action])
            G += reward
            state = state_
        # if episode % 2 == 0:
        display(print('Episode {} State: {} Total Reward: {}'.format(episode,state, G)))
        env.render()
        clear_output(wait=True)
    env.env.close()