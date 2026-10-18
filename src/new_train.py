import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from buffer import ReplayBuffer
from environment import TradingEnv
from model import NashDQN

# =============================
# HELPER FUNCTIONS
# =============================

def moving_average(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode='valid')

# =============================
# TRAINING FUNCTION (MULTI-SEED)
# =============================

def train_multiple_seeds(env_class, model_class, seeds=1, episodes=500, batch_size=64, gamma=0.99, lambda_psi=1e-3):

    all_rewards = []
    all_losses = []

    for seed in range(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = env_class()
        model = model_class()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        target_model = model_class().to(device)
        target_model.load_state_dict(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        buffer = ReplayBuffer(max_size=10000)

        rewards = []
        losses = []

        for ep in range(episodes):

            state = env.reset()
            total_reward = 0

            # Noise schedule
            noise_std = max(0.01, 0.1 * (0.995 ** ep))

            for t in range(env.M):
                x = state
                inv = env.inv.copy()

                x_tensor = torch.FloatTensor(x).unsqueeze(0)
                inv_tensor = torch.FloatTensor(inv).unsqueeze(0)

                with torch.no_grad():
                    action = model.nash_action(x_tensor, inv_tensor)
                action = action.numpy()[0]

                # Exploration noise
                noise = np.random.normal(0, noise_std, size=2)
                action = np.clip(action + noise, -2, 2)

                next_state, reward, done = env.step(action)
                next_inv = env.inv.copy()

                buffer.add(state, inv, action, reward, next_state, next_inv, done)
                state = next_state
                total_reward += reward[0]

                # Training step
                if len(buffer) > batch_size:
                    states, invs, actions, rewards_b, next_states, next_invs, dones = buffer.sample(batch_size)

                    s = torch.FloatTensor(states)
                    inv = torch.FloatTensor(invs)
                    a = torch.FloatTensor(actions)
                    r = torch.FloatTensor(rewards_b)
                    s2 = torch.FloatTensor(next_states)
                    inv2 = torch.FloatTensor(next_invs)
                    d = torch.FloatTensor(dones)

                    Q, psi = model(s, inv, a)
                    with torch.no_grad():
                        next_actions = target_model.nash_action(s2, inv2)
                        Q_next, _ = target_model(s2, inv2, next_actions)
                        y = r + gamma * Q_next * (1 - d.unsqueeze(1))

                    loss_td = ((Q - y) ** 2).mean()
                    loss_psi = psi.abs().mean()
                    loss = loss_td + lambda_psi * loss_psi

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item())

            rewards.append(total_reward)

            # Update target
            if ep % 10 == 0:
                target_model.load_state_dict(model.state_dict())
                print(f"[Seed {seed}] Episode {ep}, Reward: {total_reward:.3f}, Noise: {noise_std:.3f}")

        all_rewards.append(rewards)
        all_losses.append(losses)

    return np.array(all_rewards), np.array(all_losses)

# =============================
# TRAINING
# =============================

seeds = 5
episodes = 500
batch_size = 64

all_rewards, all_losses = train_multiple_seeds(
    TradingEnv, NashDQN, seeds=seeds, episodes=episodes, batch_size=batch_size
)

# =============================
# PLOTS
# =============================

# Rewards
mean_rewards = all_rewards.mean(axis=0)
std_rewards = all_rewards.std(axis=0)

plt.figure(figsize=(10,5))
plt.plot(mean_rewards, label="Mean Reward")
plt.fill_between(range(episodes), mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, label="Std Dev")
plt.plot(moving_average(mean_rewards), label="Moving Average", linestyle='--')
plt.title("Episode Rewards over Multiple Seeds")
plt.xlabel("Episodes")
plt.ylabel("Total Reward (Agent 1)")
plt.legend()
plt.grid()
plt.show()

# Losses (flatten all steps)
all_losses_flat = [item for sublist in all_losses for item in sublist]
plt.figure(figsize=(10,5))
plt.plot(moving_average(all_losses_flat, window=50))
plt.title("Training Loss (Moving Average)")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.grid()
plt.show()

