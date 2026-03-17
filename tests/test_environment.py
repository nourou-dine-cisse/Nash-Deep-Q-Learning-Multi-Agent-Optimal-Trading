import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import TradingEnv
import numpy as np
import matplotlib.pyplot as plt

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
    i += 1

import numpy as np

env = TradingEnv()

# Disable all interactions — pure Brownian motion
env.kappa = 0.0
env.rho   = 0.0
env.eta   = 0.0
env.gamma = 0.0

final_prices = []
sample_paths = []          # store a few trajectories for the path plot
N_PATHS       = 10000
N_SAMPLE_PLOT = 20         # number of trajectories to display

for ep in range(N_PATHS):
    env.reset()
    path = [env.S]         # record initial price
    for _ in range(env.M):
        env.step(np.zeros(env.N))
        path.append(env.S)
    final_prices.append(env.S)
    if ep < N_SAMPLE_PLOT:
        sample_paths.append(path)

var      = np.var(final_prices)
expected = env.sigma**2 * env.T   # = 0.01^2 * 5 = 0.0005

print(f"Var(S_T)  = {var:.6f}")
print(f"Expected  = {expected:.6f}")
print(f"Ratio     = {var/expected:.3f}")  # should be between 0.95 and 1.05

# --- Plot 1: sample Brownian motion paths ---------------------------------
timesteps = np.linspace(0, env.T, env.M + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Brownian Motion Validation (kappa=rho=eta=gamma=0)", fontsize=13)

ax1 = axes[0]
for path in sample_paths:
    ax1.plot(timesteps, path, alpha=0.5, linewidth=0.8)
ax1.set_title(f"{N_SAMPLE_PLOT} Sample Price Paths")
ax1.set_xlabel("Time t")
ax1.set_ylabel("Price S(t)")
ax1.grid(True, alpha=0.3)

# --- Plot 2: histogram of final prices vs theoretical Gaussian -------------
ax2 = axes[1]
ax2.hist(final_prices, bins=80, density=True, color="steelblue",
         alpha=0.7, label="Simulated S(T)")

# overlay theoretical normal density N(S0, sigma^2 * T)
S0  = sample_paths[0][0]
std = np.sqrt(expected)
xs  = np.linspace(min(final_prices), max(final_prices), 300)
pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - S0) / std) ** 2)
ax2.plot(xs, pdf, color="crimson", linewidth=2,
         label="N(S0, sigma^2 * T)")  # theoretical distribution

ax2.set_title(f"Distribution of S(T) over {N_PATHS:,} episodes")
ax2.set_xlabel("Final Price S(T)")
ax2.set_ylabel("Density")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("brownian_validation.png", dpi=150)
plt.show()