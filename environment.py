import torch
import numpy as np
from model import *

class TradingEnv:
    """
    Market simulator for the 2-agent optimal execution problem.
    Implements the stochastic game from Casgrain, Ning & Jaimungal (2022).

    State vector (eq. 20): x_t = [S_t/S_0, t/T, total_sold/total_inv0, Y_t]
    """

    def __init__(self):
        # --- Market parameters (Table 1 of paper) ---
        self.kappa = 0.1    # mean-reversion speed of price toward theta
        self.theta = 10.0   # long-run mean price level
        self.sigma = 0.01   # price volatility (diffusion coefficient)
        self.gamma = 0.02   # transient impact scale (NOT the discount factor)
        self.rho   = 0.5    # transient impact decay rate
        self.eta   = 0.05   # permanent price impact coefficient
        self.b1    = 0.1    # linear transaction cost coefficient
        self.b2    = 0.1    # terminal inventory penalty coefficient
        self.b3    = 0.0    # running inventory penalty (zero in Table 1)

        # --- Simulation settings ---
        self.N  = 2               # number of agents
        self.T  = 5.0             # time horizon in hours
        self.M  = 10             # number of decision steps
        self.dt = self.T / self.M # time step size = 0.5

    # ------------------------------------------------------------------

    def reset(self):
        """
        Reset all environment variables for a new episode.
        Must be called before every training episode.

        Returns:
            np.array shape (4,): initial state vector x_0
        """
        self.S    = 10.0                 # initial mid-price
        self.Y    = 0.0                  # transient impact starts at zero
        self.t    = 0.0                  # current time
        self.inv  = np.ones(self.N)      # each agent starts with inventory = 1
        self.inv0 = self.inv.copy()      # store initial inventory for _get_state()

        return self._get_state()

    # ------------------------------------------------------------------

    def step(self, actions):
        """
        Advance the environment by one timestep given all agents' actions.

        Update order: reward → inventory → time → Y → S
        Reward must use S_t BEFORE the price update (reward is for trading at current price).

        Args:
            actions: array-like shape (N,) — trading rates nu_i
                     positive = buying, negative = selling

        Returns:
            next_state : np.array shape (4,)
            rewards    : np.array shape (N,)
            done       : bool
        """
        nu    = np.array(actions)
        nu_bar = np.sum(nu)         # aggregate flow — drives price impact (eqs. 14, 15)

        # Step 1 — Reward uses current S and current inv (before any updates)
        rewards = self._compute_reward(nu)

        # Step 2 — Update inventory: q_{i,t+dt} = q_{i,t} + nu_i * dt  (eq. 17)
        self.inv = self.inv + nu * self.dt

        # Step 3 — Advance time
        self.t += self.dt

        # Step 4 — Update transient impact Y (eq. 15)
        Y_old  = self.Y
        h      = np.sign(nu_bar) * np.sqrt(np.abs(nu_bar) + 1e-8)  # sqrt impact, safe near 0
        self.Y = (1 - self.rho * self.dt) * self.Y + self.gamma * h * self.dt
        dY     = self.Y - Y_old     # change in Y for this step, needed for S update

        # Step 5 — Update mid-price S (eq. 14) — Euler-Maruyama discretisation
        # dS = [kappa*(theta - S) + eta*nu_bar]*dt + dY + sigma*sqrt(dt)*epsilon
        drift  = self.kappa * (self.theta - self.S) + self.eta * nu_bar
        noise  = self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.S = self.S + drift * self.dt + dY + noise

        done = self.t >= self.T - 1e-9
        return self._get_state(), rewards, done

    # ------------------------------------------------------------------

    def _compute_reward(self, nu):
        """
        Per-agent reward for the current timestep (eq. 19).

        Three components:
          (i)   Trading cost:          -nu_i * (S + b1*nu_i) * dt
          (ii)  Running urgency:       -b3 * q_i^2 * dt          (zero when b3=0)
          (iii) Terminal liquidation:   q_{i,T} * (S_T - b2*q_{i,T})  — last step only

        The terminal term uses q_{i,T} = inv[i] + nu[i]*dt (inventory AFTER this trade).
        The terminal condition checks self.t against T-dt because self.t has not been
        advanced yet when this method is called.

        Args:
            nu: np.array shape (N,)

        Returns:
            rewards: np.array shape (N,)
        """
        rewards = np.zeros(self.N)

        is_terminal = self.t >= self.T - self.dt - 1e-9  # float-safe last-step check

        for i in range(self.N):
            trading_cost = -nu[i] * (self.S + self.b1 * nu[i]) * self.dt
            urgency_cost = -self.b3 * (self.inv[i] ** 2) * self.dt
            rewards[i]   = trading_cost + urgency_cost

            if is_terminal:
                q_T           = self.inv[i] + nu[i] * self.dt  # inventory after final trade
                terminal_gain = q_T * (self.S - self.b2 * q_T)
                rewards[i]   += terminal_gain

        return rewards

    # ------------------------------------------------------------------

    def _get_state(self):
        """
        Build the 4-dimensional state vector (eq. 20).

        Components:
          [0] S_t / S_0          — normalised price (S_0 = 10)
          [1] t / T              — fraction of horizon elapsed
          [2] total_sold / sum(inv0) — normalised aggregate inventory change
          [3] Y_t                — transient impact level (already small scale)

        Returns:
            np.array shape (4,), dtype float32
        """
        total_sold = np.sum(self.inv0 - self.inv)

        return np.array([
            self.S / 10.0,
            self.t / self.T,
            total_sold / np.sum(self.inv0),
            self.Y
        ], dtype=np.float32)
        

class TWAPAgent:
    """
    Time Weighted Average Price baseline.
    Sells inventory uniformly across all time steps.
    nu_i = q_{i,0} / T  (constant selling rate)
    """
    
    def __init__(self, env):
        self.rate = env.inv0[0] / env.T   # = 1.0 / 5.0 = 0.2
    
    def act(self, state):
        # ignore state completely always same action
        return -self.rate   # negative = selling


class NashPolicy:
    """
    Wraps NashDQN to expose the same act(state) interface as TWAPAgent.
    Used by fictitious_play after iteration 1.
    """
    def __init__(self, nash_dqn, env):
        self.nash_dqn = nash_dqn
        self.env      = env

    def act(self, state):
        x   = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, 4)
        inv = torch.tensor(self.env.inv[:2], dtype=torch.float32).unsqueeze(0)  # (1, 2)
        with torch.no_grad():
            return self.nash_dqn.nash_action(x, inv)[0, 1].item()    
    
def compute_best_response(state, inv, u_other, adv_net):
    """
    Analytical best-response of agent 0 given opponent plays u_other.
    Derived from dA/du_1 = 0 using AdvantageNet outputs.
    """
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)   # (1, 4)
    inventories = torch.tensor(inv[:2], dtype=torch.float32).unsqueeze(0)   # (1, 2)

    mu, P11, P12, P22, psi = adv_net(x, inventories)

    # dA_1/du_1 = -2*P11*(u1-mu1) - P12*(u2-mu2) = 0
    u1_star = mu[:, 0] - (P12[:, 0] * (u_other - mu[:, 1])) / (2 * P11[:, 0])

    return u1_star.item()
def fictitious_play(env, nash_dqn, B=10):

    policy_others = TWAPAgent(env)   # Init : TWAP as initial policy

    for b in range(B):
        state = env.reset()
        done  = False

        while not done:
            u_other        = policy_others.act(state)
            u0             = compute_best_response(state, env.inv, u_other, nash_dqn.adv_net)
            state, _, done = env.step(np.array([u0, u_other]))

        # symmetry — opponent adopts Nash policy for next iteration
        policy_others = NashPolicy(nash_dqn, env)

    return policy_others