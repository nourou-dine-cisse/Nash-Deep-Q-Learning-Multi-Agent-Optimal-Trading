from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer storing (s, a, r, s', done) transitions.
    Decorrelates gradient updates by sampling random minibatches (Algorithm 5.1).
    """

    def __init__(self, max_size=100_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        """
        Store one transition as a tuple.

        Args:
            state      : np.array (4,)
            action     : np.array (N,)
            reward     : np.array (N,)
            next_state : np.array (4,)
            done       : bool
        """
        self.buffer.append((state, action, reward, next_state, done))
        return

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

    def __len__(self):
        # lets you write: if len(buffer) > min_size
        return len(self.buffer)