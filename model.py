import torch
import torch.nn as nn
from collections import deque, namedtuple
import numpy as np
from buffer import ReplayBuffer
import random


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def _mlp(sizes: list, activate_last: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = (i == len(sizes) - 2)
        if not is_last or activate_last:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────
# TASK 1a — Skeleton ValueNet
# Just the shape matters this week.
# Full quadratic structure comes in Week 2.
# ─────────────────────────────────────────────────────────────

class ValueNet(nn.Module):
    """
    Skeleton V̂(x).
    Input  : state x  of shape (B, 4)
    Output : scalar   of shape (B, 1)
    Paper Section 6.2: 4 hidden layers x 32 nodes, SiLU.
    """

    def __init__(self, d_state: int = 4, hidden: int = 32):
        super().__init__()
        self.net = _mlp([d_state, hidden, hidden, hidden, hidden, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 4)  →  (B, 1)
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# TASK 1b — Skeleton AdvantageNet
# This week: just outputs 5 random numbers per agent.
# The quadratic structure (P11, P12, P22, mu, psi) comes Week 2.
# ─────────────────────────────────────────────────────────────

class AdvantageNet(nn.Module):
    """
    Skeleton Â(x; u).
    Input  : state x (B, 4) + inventories (B, N)
    Output : 5 raw scalars (B, 5)  — will become (mu, L11, P12, P22, psi) in Week 2
    Paper Section 6.2: 4 hidden layers x 32 nodes, SiLU.
    """

    def __init__(self, d_state: int = 4, n_agents: int = 5,
                 hidden: int = 32, perm_out: int = 32):
        super().__init__()
        self.n_agents = n_agents

        # Will be filled properly in Week 2 with PermutationInvariantLayer
        # For now: just a plain MLP on the full state
        self.net  = _mlp([d_state + n_agents, hidden, hidden, hidden, hidden],
                         activate_last=True)
        self.head = nn.Linear(hidden, 5)
        # 5 outputs = (mu, L11_raw, P12, P22, psi) — used fully in Week 2

    def forward(self, x: torch.Tensor,
                inventories: torch.Tensor) -> torch.Tensor:
        # x           : (B, 4)
        # inventories : (B, N)
        combined = torch.cat([x, inventories], dim=-1)  # (B, 4+N)
        h   = self.net(combined)   # (B, hidden)
        raw = self.head(h)         # (B, 5)
        return raw                 # raw random outputs this week — shapes only


# ─────────────────────────────────────────────────────────────
# TASK 4 — Permutation-Invariant Layer    eq. (4.10)
# f_inv(z) = σ( Σ_j φ(z_j) )
# ─────────────────────────────────────────────────────────────

class PermutationInvariantLayer(nn.Module):
    """
    Implements eq. (4.10) from the paper.

    φ is applied to each agent's features independently (shared weights).
    Results are SUMMED over agents — this sum is what makes the output
    invariant to any relabelling of agents.
    σ then transforms the aggregate.

    Paper Section 6.2: φ has 3 hidden layers x 20 nodes, SiLU.
    """

    def __init__(self, d_in: int = 1, d_hidden: int = 20,
                 d_out: int = 32, n_agents: int = 2):
        super().__init__()
        self.n_agents = n_agents

        # φ : shared across all agents
        self.phi   = _mlp([d_in, d_hidden, d_hidden, d_hidden],
                          activate_last=True)

        # σ : transforms the summed aggregate
        self.sigma = _mlp([d_hidden, d_out], activate_last=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z   : (B, N, d_in)   one feature per agent
        out : (B, d_out)     permutation-invariant aggregate
        """
        phi_out = self.phi(z)          # (B, N, d_hidden)
        agg     = phi_out.sum(dim=1)   # (B, d_hidden)  ← the key invariant step
        return   self.sigma(agg)       # (B, d_out)



