import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import TradingEnv
import numpy as np

env = TradingEnv()
state = env.reset()
done = False

i = 0
while not done:
	actions = np.random.uniform(-1, 1, size=env.N)  # random actions for testing
	next_state, rewards, done = env.step(actions)
	print(f"Next state: {i}", next_state)
	print("Rewards:", rewards)
	print("Done:", done)
	i+=1
 
import numpy as np

env = TradingEnv()

# Disable all interactions — pure Brownian motion
env.kappa = 0.0
env.rho   = 0.0
env.eta   = 0.0
env.gamma = 0.0

final_prices = []
for _ in range(10_000):
    env.reset()
    for _ in range(env.M):
        env.step(np.zeros(env.N))
    final_prices.append(env.S)  # raw price, not normalized

var      = np.var(final_prices)
expected = env.sigma**2 * env.T   # = 0.01^2 * 5 = 0.0005

print(f"Var(S_T)  = {var:.6f}")
print(f"Expected  = {expected:.6f}")
print(f"Ratio     = {var/expected:.3f}")  # doit être entre 0.95 et 1.05