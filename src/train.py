import numpy as np
import torch
import torch.optim as optim

# IMPORTANT: using your existing buffer + models
from buffer import ReplayBuffer
from environment import *
from model import *

# =============================
# TRAINING FUNCTION (ADAPTED)
# =============================

def train(env, model, episodes=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    target_model = type(model)().to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    buffer = ReplayBuffer(max_size=10000)

    gamma = 0.99
    lambda_psi = 1e-3

    losses = []
    rewards = []

    for ep in range(episodes):

        state = env.reset()
        total_reward = 0

        for t in range(env.M):
            x = state
            inv = env.inv.copy()
            
            x_tensor = torch.FloatTensor(x).unsqueeze(0)
            inv_tensor = torch.FloatTensor(inv).unsqueeze(0)
            
            with torch.no_grad():
                action = model.nash_action(x_tensor, inv_tensor)
            action = action.numpy()[0]
            
            noise_std = max(0.01, 0.1 * (0.995 ** ep))

            noise = np.random.normal(0, noise_std, size=2)

            action = action + noise # Exploration
            action = np.clip(action, -2, 2) # limiting the action
            
            next_state, reward, done = env.step(action) # update the environment
            next_inv = env.inv.copy()
            
            buffer.add(state, inv, action, reward, next_state, next_inv, done) # stocking the variables
            state = next_state # updating the state
            total_reward += reward[0]
            
            if len(buffer) > batch_size:
                states, invs, actions, rewards, next_states, next_invs, dones = buffer.sample(batch_size)
                s    = torch.FloatTensor(states)
                inv  = torch.FloatTensor(invs)
                a    = torch.FloatTensor(actions)
                r    = torch.FloatTensor(rewards)
                s2   = torch.FloatTensor(next_states)
                inv2 = torch.FloatTensor(next_invs)
                d    = torch.FloatTensor(dones)
                
                Q, psi = model(s, inv, a) # Computing
            
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

        np.append(rewards, total_reward)

        if ep % 10 == 0:
            target_model.load_state_dict(model.state_dict())
            print(f"Episode {ep}, Reward: {total_reward}")

    return model, losses, rewards
            
import matplotlib.pyplot as plt

# -------------------------
# INITIALISATION
# -------------------------
env = TradingEnv()
model = NashDQN()

# -------------------------
# TRAINING
# -------------------------
model, losses, rewards = train(
    env=env,
    model=model,
    episodes=500,
    batch_size=64
)

# -------------------------
# PLOT 1 : LOSS
# -------------------------
plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.grid()
plt.show()

# -------------------------
# PLOT 2 : REWARD
# -------------------------
plt.figure()
plt.plot(rewards)
plt.title("Episode Rewards (Agent 1)")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.grid()
plt.show()

def plot_inventory(env, model):

    state = env.reset()

    q1_hist = []
    q2_hist = []
    t_hist = []

    for t in range(env.M):

        x = state
        inv = env.inv.copy()

        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        inv_tensor = torch.FloatTensor(inv).unsqueeze(0)

        with torch.no_grad():
            action = model.nash_action(x_tensor, inv_tensor)

        action = action.numpy()[0]

        q1_hist.append(env.inv[0])
        q2_hist.append(env.inv[1])
        t_hist.append(env.t)

        state, _, done = env.step(action)

        if done:
            break

    plt.figure()
    plt.plot(t_hist, q1_hist, label="Agent 1")
    plt.plot(t_hist, q2_hist, label="Agent 2")
    plt.title("Inventory Trajectory")
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.legend()
    plt.grid()
    plt.show()

def plot_price(env, model):

    state = env.reset()

    S_hist = []
    t_hist = []

    for t in range(env.M):

        x = state
        inv = env.inv.copy()

        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        inv_tensor = torch.FloatTensor(inv).unsqueeze(0)

        with torch.no_grad():
            action = model.nash_action(x_tensor, inv_tensor)

        action = action.numpy()[0]

        S_hist.append(env.S)
        t_hist.append(env.t)

        state, _, done = env.step(action)

        if done:
            break

    plt.figure()
    plt.plot(t_hist, S_hist)
    plt.title("Price Trajectory")
    plt.xlabel("Time")
    plt.ylabel("S(t)")
    plt.grid()
    plt.show()

plot_inventory(env, model)
plot_price(env, model)