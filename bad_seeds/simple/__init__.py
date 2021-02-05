from functools import partial

import numpy as np

from tensorforce.environments import Environment





class BadSeeds(Environment):
    """

    """

    def __init__(self, seed_count=10, bad_seed_count=3, max_episode_timesteps=100):
        super().__init__()

        self.seed_count = seed_count
        self.bad_seed_count = bad_seed_count
        self._max_episode_timesteps = max_episode_timesteps

        self.rng = np.random.default_rng()

        self.bad_seeds = None
        self.good_seeds = None
        self.all_seeds = None
        self.bad_seed_indices = None
        self.good_seed_indices = None
        self.turn = None
        self.state = None

        self.reset()

    def states(self):
        return dict(
            type="float", shape=(self._max_episode_timesteps, len(self.all_seeds))
        )

    def actions(self):
        return dict(type="int", num_values=len(self.all_seeds))

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; restrict training timesteps via
    #     Environment.create(..., max_episode_timesteps=???)
    # def max_episode_timesteps(self):
    #    # TODO: figure out how to do this correctly
    #    return self._max_episode_timesteps

    # Optional additional steps to close environment
    # def close(self):
    #     super().close()

    def reset(self):
        self.bad_seeds = [
            partial(self.rng.normal, loc=0.0, scale=10.0)
            for _ in range(self.bad_seed_count)
        ]
        self.good_seeds = [
            partial(self.rng.normal, loc=0.0, scale=1.0)
            for _ in range(self.seed_count - self.bad_seed_count)
        ]

        self.all_seeds = self.bad_seeds + self.good_seeds
        self.rng.shuffle(self.all_seeds)

        self.bad_seed_indices = [
            self.all_seeds.index(bad_seed) for bad_seed in self.bad_seeds
        ]
        self.good_seed_indices = [
            self.all_seeds.index(good_seed) for good_seed in self.good_seeds
        ]

        self.turn = 0

        # max_turns x N
        self.state = np.zeros((self._max_episode_timesteps, len(self.all_seeds)))
        return self.state

    def execute(self, actions):
        seed_index = actions
        seed_measurement = self.all_seeds[seed_index](size=1)
        self.state[self.turn, seed_index] = seed_measurement
        next_state = self.state

        if self.turn < (self._max_episode_timesteps - 1):
            terminal = False
            reward = 0.0
        else:
            terminal = True

            measured_seeds = np.sum(self.state, axis=0) > 0.0
            if np.all(measured_seeds):
                reward = 100.0
                measurement_indices = np.zeros_like(self.state)
                measurement_indices[self.state > 0.0] = 1.0
                measurement_counts = np.sum(measurement_indices, axis=0, keepdims=True)
                if np.all(
                    measurement_counts[0:, self.bad_seed_indices]
                    > measurement_counts[0:, self.good_seed_indices].T
                ):
                    reward = 200.0
            else:
                reward = 0.0

        self.turn += 1

        return next_state, terminal, reward
