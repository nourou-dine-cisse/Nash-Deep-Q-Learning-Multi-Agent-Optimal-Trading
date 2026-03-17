import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import TradingEnv
from environment import TWAPAgent
import numpy as np


env = TradingEnv()
env.reset()
twap = TWAPAgent(env)
N_EPISODES = 500

rewards_twap = []

for _ in range(N_EPISODES):
    state    = env.reset()
    ep_reward = np.zeros(env.N)
    
    for _ in range(env.M):
        actions           = np.array([twap.act(state)] * env.N)
        state, rewards, done = env.step(actions)
        ep_reward        += rewards
    
    rewards_twap.append(ep_reward.mean())

print(f"TWAP mean reward : {np.mean(rewards_twap):.4f}")
print(f"TWAP std         : {np.std(rewards_twap):.4f}")