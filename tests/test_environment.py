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
timestep_rewards = []   # collect rewards of all agents at each timestep

while not done:
    actions = np.random.uniform(-1, 1, size=env.N)  # random actions for testing
    next_state, rewards, done = env.step(actions)
    print(f"Next state: {i}", next_state)
    print("Rewards:", rewards)
    print("Done:", done)
    timestep_rewards.append(rewards.copy())   # shape (N,) per timestep
    i += 1

timestep_rewards = np.array(timestep_rewards)  # (M, N)

# Plot: reward of all agents across timesteps
plt.figure(figsize=(8, 4))
for agent_i in range(env.N):
    plt.plot(timestep_rewards[:, agent_i], marker='o', linewidth=1.5,
             label=f"Agent {agent_i}")
plt.title("All Agents Reward per Timestep (random actions)")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("all_agents_rewards.png", dpi=150)
plt.show()


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

# Plot: sample Brownian motion paths
timesteps = np.linspace(0, env.T, env.M + 1)

plt.figure(figsize=(8, 4))
for path in sample_paths:
    plt.plot(timesteps, path, alpha=0.5, linewidth=0.8)
plt.title(f"{N_SAMPLE_PLOT} Sample Price Paths (kappa=rho=eta=gamma=0)")
plt.xlabel("Time t")
plt.ylabel("Price S(t)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("brownian_paths.png", dpi=150)
plt.show()