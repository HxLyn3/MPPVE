import numpy as np


class ReplayBuffer:
    """ replay buffer """
    def __init__(self, buffer_size, obs_shape, action_dim):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.memory = {
            "s":    np.zeros((buffer_size, *self.obs_shape), dtype=np.float32),
            "a":    np.zeros((buffer_size, self.action_dim), dtype=np.float32),
            "r":    np.zeros((buffer_size, 1), dtype=np.float32),
            "s_":   np.zeros((buffer_size, *self.obs_shape), dtype=np.float32),
            "done": np.zeros((buffer_size, 1), dtype=np.float32),
        }

        self.capacity = buffer_size
        self.size = 0
        self.cnt = 0

    def store(self, s, a, r, s_, done, timeout):
        """ store transition (s, a, r, s_, done) """
        done *= (1-timeout)
        self.memory["s"][self.cnt] = s
        self.memory["a"][self.cnt] = a
        self.memory["r"][self.cnt] = r
        self.memory["s_"][self.cnt] = s_
        self.memory["done"][self.cnt] = done

        self.cnt = (self.cnt+1)%self.capacity
        self.size = min(self.size+1, self.capacity)

    def store_batch(self, s, a, r, s_, done):
        """ store batch transitions (s, a, r, s_, done) """
        batch_size = len(s)

        indices = np.arange(self.cnt, self.cnt+batch_size)%self.capacity
        self.memory["s"][indices] = s
        self.memory["a"][indices] = a
        self.memory["r"][indices] = r
        self.memory["s_"][indices] = s_
        self.memory["done"][indices] = done

        self.cnt = (self.cnt+batch_size)%self.capacity
        self.size = min(self.size+batch_size, self.capacity)

    def sample(self, batch_size):
        """ sample a batch of transitions """
        indices = np.random.randint(0, self.size, batch_size)
        return {
            "s":    self.memory["s"][indices].copy(),
            "a":    self.memory["a"][indices].copy(),
            "r":    self.memory["r"][indices].copy(),
            "s_":   self.memory["s_"][indices].copy(),
            "done": self.memory["done"][indices].copy()
        }

    def sample_all(self):
        """ sample all transitions """
        indices = np.arange(self.size)
        return {
            "s":    self.memory["s"][indices].copy(),
            "a":    self.memory["a"][indices].copy(),
            "r":    self.memory["r"][indices].copy(),
            "s_":   self.memory["s_"][indices].copy(),
            "done": self.memory["done"][indices].copy()
        }


class ReplayBufferForSeqSampling(ReplayBuffer):
    """ replay buffer for sequential actions sampling """
    def __init__(self, buffer_size, obs_shape, action_dim, plan_length, gamma):
        super().__init__(buffer_size, obs_shape, action_dim)
        # used for mbpc-based policy
        self.endpoint = np.zeros(buffer_size, dtype=np.float32)     # whether the step is an endpoint (end ≠ done)
        self.sample_sign = np.zeros(buffer_size, dtype=np.float32)  # whether the step can be sampled
        self.sample_mask = np.zeros((buffer_size, plan_length), dtype=np.float32)
        self.sample_end = np.zeros(buffer_size, dtype=np.int64)
        self.plan_length = plan_length
        self.gammas = gamma**np.arange(plan_length).reshape((plan_length, 1))

    def store(self, s, a, r, s_, done, timeout):
        self.endpoint[self.cnt] = done
        self.sample_sign[self.cnt] = 0
        self.sample_mask[self.cnt] = 0
        self.sample_end[self.cnt] = 0
        super().store(s, a, r, s_, done, timeout)

        if self.size >= self.plan_length:
            if self.endpoint[np.arange(self.cnt-self.plan_length, self.cnt-1)].sum() == 0:
                self.sample_sign[self.cnt-self.plan_length] = 1
                self.sample_mask[self.cnt-self.plan_length] = 1
                self.sample_end[self.cnt-self.plan_length] = self.plan_length - 1

            elif self.memory["done"][np.arange(self.cnt-self.plan_length, self.cnt-1)].sum() == 1:
                for i in range(self.plan_length-1):
                    if self.memory["done"][self.cnt-self.plan_length+i]:
                        self.sample_sign[self.cnt-self.plan_length] = 1
                        self.sample_mask[self.cnt-self.plan_length, :i+1] = 1
                        self.sample_end[self.cnt-self.plan_length] = i
                        break

    def sample_nstep(self, batch_size):
        """ sample a batch of {plan_length}-step transitions """
        all_start_indices = np.arange(self.size)[self.sample_sign[:self.size]==1]
        start_indices = np.random.choice(all_start_indices, batch_size)
        indices = (start_indices.reshape(-1, 1) + np.arange(self.plan_length))%self.size
        sample_mask = self.sample_mask[start_indices]
        sample_end = self.sample_end[start_indices]

        return {
            "s":    self.memory["s"][start_indices].copy(),
            "a":    (self.memory["a"][indices].reshape((batch_size, -1))*sample_mask.repeat(self.action_dim, axis=-1)).copy(),
            "r":    (self.memory["r"][indices].reshape((batch_size, -1))*sample_mask).dot(self.gammas).copy(),
            "s_":   self.memory["s_"][indices[np.arange(batch_size), sample_end]].copy(),
            "done": self.memory["done"][indices].sum(axis=1).clip(None, 1).copy()
        }

    def sample_all_nstep(self):
        """ sample all {plan_length}-step transitions """
        start_indices = np.arange(self.size)[self.sample_sign[:self.size]==1]
        indices = (start_indices.reshape(-1, 1) + np.arange(self.plan_length))%self.size
        sample_mask = self.sample_mask[start_indices]
        sample_end = self.sample_end[start_indices]

        return {
            "s":    self.memory["s"][start_indices].copy(),
            "a":    (self.memory["a"][indices].reshape((indices.shape[0], -1))*sample_mask.repeat(self.action_dim, axis=-1)).copy(),
            "r":    (self.memory["r"][indices].reshape((indices.shape[0], -1))*sample_mask).dot(self.gammas).copy(),
            "s_":   self.memory["s_"][indices[np.arange(indices.shape[0]), sample_end]].copy(),
            "done": self.memory["done"][indices].sum(axis=1).clip(None, 1).copy()
        }

    def sample_nstep4rollout(self, batch_size):
        """ sample a batch of {plan_length-1}-step transitions for rollout """
        all_start_indices = np.arange(self.size)[self.sample_end[:self.size]==self.plan_length-1]
        start_indices = np.random.choice(all_start_indices, batch_size)
        indices = (start_indices.reshape(-1, 1) + np.arange(self.plan_length-1))%self.size

        return {
            "s":    self.memory["s"][indices].reshape((batch_size, -1)).copy(),
            "a":    self.memory["a"][indices].reshape((batch_size, -1)).copy(),
            "r":    self.memory["r"][indices].reshape((batch_size, -1)).copy(),
            "s_":   self.memory["s_"][indices].reshape((batch_size, -1)).copy(),
            "done": self.memory["done"][indices].reshape((batch_size, -1)).copy()
        }
