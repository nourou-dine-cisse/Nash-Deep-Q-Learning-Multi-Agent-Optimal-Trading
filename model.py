"""
model.py — Student D, Week 2
==============================
Builds on Week 1 and adds:
  - Real quadratic structure inside AdvantageNet
  - Cholesky enforcement P11 > 0 at all times
  - compute_advantage() implementing eq. (4.9) of the paper
  - NashDQN: combines ValueNet + AdvantageNet -> Q(x, u)
  - nash_action(): analytic Nash equilibrium action eq. (4.6)

Reference: arXiv:1904.10554v2 — Casgrain, Ning & Jaimungal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# =============================================================================
# HELPER
# =============================================================================

def _mlp(sizes: list, activate_last: bool = False) -> nn.Sequential:
    """
    Builds a fully-connected MLP with SiLU activations between layers.

    Args:
        sizes         : list of layer widths e.g. [4, 32, 32, 1]
        activate_last : if True, adds SiLU after the last layer too

    Example:
        _mlp([4, 32, 32, 1])
        -> Linear(4,32) + SiLU + Linear(32,32) + SiLU + Linear(32,1)
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = (i == len(sizes) - 2)
        if not is_last or activate_last:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


# =============================================================================
# WEEK 1 
# =============================================================================

# Permutation Invariant Layer

class PermutationInvariantLayer(nn.Module):
    """
    Implements equation (4.10) from the paper:

        f_inv(z) = sigma( sum_j phi(z_j) )

    phi is applied independently to each agent's features (shared weights).
    The SUM over agents makes the output invariant to any agent relabelling.
    sigma then transforms the aggregate.

    Paper Section 6.2: phi has 3 hidden layers x 20 nodes, SiLU.

    Args:
        d_in     : feature size per agent   (1 = scalar inventory)
        d_hidden : hidden width of phi      (20 in paper)
        d_out    : output dimension         (32 in paper)
        n_agents : N                        (2)
    """

    def __init__(self, d_in: int = 1, d_hidden: int = 20,
                 d_out: int = 32, n_agents: int = 2):
        super().__init__()
        self.n_agents = n_agents

        # phi: shared weights across all agents
        self.phi = _mlp(
            [d_in, d_hidden, d_hidden, d_hidden],
            activate_last=True
        )

        # sigma: transforms the summed aggregate
        self.sigma = _mlp([d_hidden, d_out], activate_last=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z   : (B, N, d_in)   one feature vector per agent

        Returns:
            out : (B, d_out)     permutation-invariant aggregate
        """
        phi_out = self.phi(z)           # (B, N, d_hidden)
        agg     = phi_out.sum(dim=1)    # (B, d_hidden)  <- invariant step
        return self.sigma(agg)          # (B, d_out)


# Value Net

class ValueNet(nn.Module):
    """
    Approximates the value function V_i(x).

    Paper Section 6.2: 4 hidden layers x 32 nodes, SiLU.
    Under identical preferences + label invariance (Section 4.1),
    V_i is the same for all agents -> single shared scalar output.

    Args:
        d_state : state dimension  (4)
        hidden  : hidden width     (32)
    """

    def __init__(self, d_state: int = 4, hidden: int = 32):
        super().__init__()
        self.net = _mlp([d_state, hidden, hidden, hidden, hidden, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x  : (B, 4)

        Returns:
            V  : (B, 1)
        """
        return self.net(x)

# =============================================================================
# WEEK 2 
# =============================================================================

# Advantage Network    A(x; u)    NEW VERSION


class AdvantageNet(nn.Module):
    """
    Outputs the parameters (mu, P11, P12, P22, psi) of the locally
    linear-quadratic advantage function.

    For N=2, scalar actions, equation (4.9) specialises to:

        A_i(x; u_i, u_j)
            = - P11_i * (u_i - mu_i)^2
              - P12_i * (u_i - mu_i) * (u_j - mu_j)
              - P22_i * (u_j - mu_j)^2
              + psi_i * (u_j - mu_j)

    Network outputs per agent: (mu_i, L11_i, P12_i, P22_i, psi_i)
    where P11_i = L11_i^2 + eps  (scalar Cholesky, always > 0)

    Architecture (Section 6.2):
        perm-inv layer (3x20)  ->  concatenated with state
        ->  main network (4x32, SiLU)
        ->  output head (5 scalars)

    Args:
        d_state     : full state dimension    (4)
        n_agents    : N                       (2)
        hidden      : main network width      (32)
        perm_hidden : phi hidden width        (20)
        perm_out    : perm-inv output dim     (32)
    """

    def __init__(self, d_state: int = 4, n_agents: int = 2,
                 hidden: int = 32, perm_hidden: int = 20, perm_out: int = 32):
        super().__init__()
        self.n_agents = n_agents

        # Perm-inv path over per-agent inventories (1 scalar each)
        self.perm_inv = PermutationInvariantLayer(
            d_in=1, d_hidden=perm_hidden,
            d_out=perm_out, n_agents=n_agents
        )

        # Main network: [state (4) | perm_inv output (32)] = 36 inputs
        self.main_net = _mlp(
            [d_state + perm_out, hidden, hidden, hidden, hidden],
            activate_last=True
        )

        # Output head: 5 scalars = (mu, L11_raw, P12, P22, psi)
        # One shared head under identical preferences + label invariance
        self.head = nn.Linear(hidden, 5)

    def forward(
        self,
        x: torch.Tensor,
        inventories: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        Args:
            x           : (B, 4)   state  (S_t, t, total_flow, Y_t)
            inventories : (B, N)   per-agent inventories q_{i,t}

        Returns (all shape (B, N)):
            mu   : Nash policy mu_i(x)       -- equilibrium action
            P11  : own quadratic coeff       > 0 always
            P12  : cross-action coeff
            P22  : opponent quadratic coeff
            psi  : opponent linear coeff     (target of L1 regulariser)
        """
        B = x.shape[0]

        # ── Step 1: permutation-invariant path ────────────────────────────
        # inventories: (B, N) -> (B, N, 1) for the perm-inv layer
        inv_feats = inventories.unsqueeze(-1)    # (B, N, 1)
        perm_out  = self.perm_inv(inv_feats)     # (B, 32)

        # ── Step 2: main network ──────────────────────────────────────────
        # Concatenate full state with perm-inv output
        combined = torch.cat([x, perm_out], dim=-1)   # (B, 36)
        h        = self.main_net(combined)             # (B, 32)
        raw      = self.head(h)                        # (B, 5)

        # ── Step 3: unpack the 5 parameters ──────────────────────────────
        mu_raw  = raw[:, 0]    # (B,)
        L11_raw = raw[:, 1]    # (B,)  raw value, made positive below
        P12_raw = raw[:, 2]    # (B,)
        P22_raw = raw[:, 3]    # (B,)
        psi_raw = raw[:, 4]    # (B,)

        # ── Step 4: enforce P11 > 0 via scalar Cholesky ──────────────────
        # P11 = L11^2 + eps    [Section 4 of paper]
        # Squaring guarantees P11 > 0 regardless of the raw network output.
        # This ensures A_i is concave in u_i -> Nash operator is well-defined.
        P11 = L11_raw.pow(2) + 1e-6    # (B,)  always > 0

        # ── Step 5: expand to (B, N) ──────────────────────────────────────
        # Same parameters for every agent under identical preferences
        def expand(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(1).expand(B, self.n_agents).contiguous()

        return (expand(mu_raw),  expand(P11),
                expand(P12_raw), expand(P22_raw), expand(psi_raw))


# Compute_advantage() 

def compute_advantage(
    mu:  torch.Tensor,
    P11: torch.Tensor,
    P12: torch.Tensor,
    P22: torch.Tensor,
    psi: torch.Tensor,
    u:   torch.Tensor
) -> torch.Tensor:
    """
    Evaluates A_i(x; u_1, u_2) for both agents simultaneously.

    A_i = - P11_i * (u_i - mu_i)^2
          - P12_i * (u_i - mu_i) * (u_j - mu_j)
          - P22_i * (u_j - mu_j)^2
          + psi_i * (u_j - mu_j)

    Key invariant: A_i = 0 exactly when u = mu (the Nash equilibrium point).
    This guarantees Q_i(x; mu(x)) = V_i(x) as required by eq. (4.6).

    Args:
        mu, P11, P12, P22, psi : (B, 2)  outputs from AdvantageNet
        u                      : (B, 2)  joint actions [u_1, u_2]

    Returns:
        A : (B, 2)  advantage value for each agent
    """
    # Deviation from Nash action
    delta = u - mu       # (B, 2)
    d1    = delta[:, 0]  # (B,)  agent 1 deviation
    d2    = delta[:, 1]  # (B,)  agent 2 deviation

    # Agent 1  (own=1, opponent=2)
    A1 = (- P11[:, 0] * d1.pow(2)
          - P12[:, 0] * d1 * d2
          - P22[:, 0] * d2.pow(2)
          + psi[:, 0] * d2)

    # Agent 2  (own=2, opponent=1)  <- roles swap
    A2 = (- P11[:, 1] * d2.pow(2)
          - P12[:, 1] * d2 * d1
          - P22[:, 1] * d1.pow(2)
          + psi[:, 1] * d1)

    return torch.stack([A1, A2], dim=1)    # (B, 2)


# NashDQN: full model

class NashDQN(nn.Module):
    """
    Complete Nash-DQN model.

    Q_i(x; u) = V_i(x) + A_i(x; u)    [eq. 4.3]
    u*_i(x)   = mu_i^theta(x)          [eq. 4.6]  <- analytic Nash action

    The Nash action is computed analytically from the network output.
    No inner optimisation loop needed -- that is the key advantage of
    the quadratic structure.

    Args:
        d_state     : state dimension  (4)
        n_agents    : N                (2)
        hidden      : hidden width     (32)
        perm_hidden : phi width        (20)
        perm_out    : perm-inv output  (32)
    """

    def __init__(self, d_state: int = 4, n_agents: int = 2,
                 hidden: int = 32, perm_hidden: int = 20, perm_out: int = 32):
        super().__init__()
        self.n_agents = n_agents

        self.value_net = ValueNet(d_state=d_state, hidden=hidden)

        self.adv_net = AdvantageNet(
            d_state=d_state, n_agents=n_agents,
            hidden=hidden, perm_hidden=perm_hidden, perm_out=perm_out
        )

    def forward(
        self,
        x:           torch.Tensor,
        inventories: torch.Tensor,
        u:           torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Q_i(x; u) = V_i(x) + A_i(x; u).

        Args:
            x           : (B, 4)
            inventories : (B, 2)
            u           : (B, 2)  joint actions [u_1, u_2]

        Returns:
            Q   : (B, 2)  Q-value per agent
            psi : (B, 2)  psi values needed for L1 regulariser in loss
        """
        # V(x) expanded to match both agents
        V = self.value_net(x).expand(-1, self.n_agents)    # (B, 2)

        # Advantage parameters from network
        mu, P11, P12, P22, psi = self.adv_net(x, inventories)

        # Advantage value
        A = compute_advantage(mu, P11, P12, P22, psi, u)   # (B, 2)

        return V + A, psi

    def nash_action(
        self,
        x:           torch.Tensor,
        inventories: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns u*(x) = mu^theta(x).
        Analytic closed form -- no inner loop.    [eq. 4.6]

        Args:
            x           : (B, 4)
            inventories : (B, 2)

        Returns:
            mu : (B, 2)  Nash equilibrium actions for both agents
        """
        mu, _, _, _, _ = self.adv_net(x, inventories)
        return mu