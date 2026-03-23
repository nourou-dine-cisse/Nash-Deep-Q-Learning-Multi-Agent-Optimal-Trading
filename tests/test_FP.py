import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from environment import TradingEnv, TWAPAgent, fictitious_play
from model import NashDQN

env      = TradingEnv()
nash_dqn = NashDQN(d_state=4, n_agents=2)

# Test 1 — TWAPAgent interface
state = env.reset()
twap  = TWAPAgent(env)
assert twap.act(state) == -0.2, "TWAPAgent rate wrong"
print("TWAPAgent OK — rate:", twap.act(state))

# Test 2 — compute_best_response runs without crash
from environment import compute_best_response
u0 = compute_best_response(state, env.inv, -0.2, nash_dqn.adv_net)
assert isinstance(u0, float), "compute_best_response should return float"
print("compute_best_response OK — u0:", round(u0, 4))

# Test 3 — fictitious_play runs B iterations without crash
policy = fictitious_play(env, nash_dqn, B=3)
assert hasattr(policy, 'act'), "returned policy must have act() method"
print("fictitious_play OK — returned policy:", policy)

# Test 4 — returned policy produces a float action
state  = env.reset()
action = policy.act(state)
assert isinstance(action, float), "policy.act() should return float"
print("policy.act() OK — action:", round(action, 4))

print("\nAll tests passed.")