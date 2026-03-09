import torch
import torch.nn as nn
from collections import deque, namedtuple
import numpy as np
from buffer import Replaybuffer
from model import mlp,ValueNet, AdvantageNet, PermutationInvariantLayer
import random

# ─────────────────────────────────────────────────────────────
# TASK 3 — Dummy data loop + shape checks
# TASK 5 — Label-invariance verification
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    B, N = 16, 2

    value_net = ValueNet(d_state=4, hidden=32)
    adv_net   = AdvantageNet(d_state=4, n_agents=2, hidden=32)
    perm_layer = PermutationInvariantLayer(d_in=1, d_hidden=20,
                                           d_out=32, n_agents=2)
    buf = ReplayBuffer(capacity=100_000)

    # ── TASK 3: dummy data collection loop — 1000 steps ──────────────────
    print("Running 1000 dummy steps...")
    for step in range(1000):
        x      = np.random.randn(4).astype(np.float32)   # state (4,)
        u      = np.random.randn(2).astype(np.float32)   # action (2,)
        r      = np.random.randn(1).astype(np.float32)   # reward (1,)
        x_next = np.random.randn(4).astype(np.float32)   # next state (4,)
        done   = False

        buf.add(x, u, r, x_next, done)

    print(f"Buffer size : {len(buf)}")
    assert len(buf) == 1000

    # Sample a batch and verify shapes
    batch = buf.sample(B)

    # Paper requires these exact shapes:
    assert batch.x.shape      == (B, 4), f"x shape wrong: {batch.x.shape}"
    assert batch.u.shape      == (B, 2), f"u shape wrong: {batch.u.shape}"
    assert batch.r.shape      == (B, 1), f"r shape wrong: {batch.r.shape}"
    assert batch.x_next.shape == (B, 4), f"x_next shape wrong: {batch.x_next.shape}"

    print(f"[OK] x      : {batch.x.shape}")       # (B, 4)
    print(f"[OK] u      : {batch.u.shape}")       # (B, 2)
    print(f"[OK] r      : {batch.r.shape}")       # (B, 1)
    print(f"[OK] x_next : {batch.x_next.shape}")  # (B, 4)

    # ── Verify network output shapes ──────────────────────────────────────
    x_batch   = batch.x                           # (B, 4)
    inv_batch = torch.rand(B, N)                  # (B, 2)  dummy inventories

    V_out = value_net(x_batch)
    assert V_out.shape == (B, 1), f"ValueNet shape wrong: {V_out.shape}"
    print(f"[OK] ValueNet output   : {V_out.shape}")   # (B, 1)

    A_out = adv_net(x_batch, inv_batch)
    assert A_out.shape == (B, 5), f"AdvantageNet shape wrong: {A_out.shape}"
    print(f"[OK] AdvantageNet output : {A_out.shape}") # (B, 5)

    # ── TASK 4+5: Permutation-invariant layer + label-invariance check ────
    inv_feats      = inv_batch.unsqueeze(-1)       # (B, N, 1)
    inv_feats_perm = inv_feats[:, [1, 0], :]       # swap agent 0 and agent 1

    out_orig = perm_layer(inv_feats)               # (B, 32)
    out_perm = perm_layer(inv_feats_perm)          # (B, 32)

    diff = (out_orig - out_perm).abs().max().item()
    assert diff < 1e-5, f"Permutation invariance failed! diff={diff}"
    print(f"[OK] Perm-inv layer output : {out_orig.shape}")   # (B, 32)
    print(f"[OK] Label-invariance diff : {diff:.2e}  (must be ~0)")

    print("\nWeek 1 checks all passed.")