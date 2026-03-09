# from buffer import ReplayBuffer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from buffer import ReplayBuffer

buf = ReplayBuffer(max_size=5)  # petit buffer pour tester l'éviction

# Remplis au-delà de max_size — les plus vieux doivent disparaître
for i in range(7):
    buf.add(np.zeros(4), np.zeros(2), np.zeros(2), np.zeros(4), False)

assert len(buf) == 5, "Buffer should evict old entries"

# Test sample shapes
states, actions, rewards, next_states, dones = buf.sample(3)
assert states.shape      == (3, 4), f"Got {states.shape}"
assert actions.shape     == (3, 2), f"Got {actions.shape}"
assert rewards.shape     == (3, 2), f"Got {rewards.shape}"
assert next_states.shape == (3, 4), f"Got {next_states.shape}"
assert dones.shape       == (3,),   f"Got {dones.shape}"

# Test sample quand buffer < batch_size
buf2 = ReplayBuffer()
buf2.add(np.zeros(4), np.zeros(2), np.zeros(2), np.zeros(4), False)
states, _, _, _, _ = buf2.sample(100)
assert states.shape == (1, 4), "Should sample everything when buffer < batch_size"

print("All buffer tests passed.")

