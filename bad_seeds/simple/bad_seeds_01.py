from functools import partial

import numpy as np
from tensorforce.environments import Environment


def count_measurements(time_steps_by_seeds_state):
    measurement_indices = np.zeros_like(time_steps_by_seeds_state)
    measurement_indices[time_steps_by_seeds_state != 0.0] = 1.0
    _measurement_counts = np.sum(measurement_indices, axis=0, keepdims=True)

    _measured_seed_count = np.sum(
        np.ones_like(_measurement_counts)[_measurement_counts > 0.0]
    )

    return _measurement_counts, _measured_seed_count


class BadSeeds01(Environment):
    def __init__(self, seed_count, bad_seed_count, max_episode_length):
        """
        I have not been able to get this to work using the standard
        max_episode_timesteps. I want to use that information in
        the states() method but it is not available at that point
        if I try
        Environment.create(
            environment=BadSeeds01,
            seed_count=10,
            bad_seed_count=3,
            max_episode_timesteps=100
        )
        """
        super().__init__()
        if bad_seed_count > seed_count:
            raise ValueError("bad_seed_count must be less than or equal to seed_count")

        self.seed_count = seed_count
        self.bad_seed_count = bad_seed_count
        self.max_episode_length = max_episode_length

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
            type="float", shape=(self.max_episode_timesteps(), len(self.all_seeds))
        )

    def actions(self):
        return dict(type="int", num_values=len(self.all_seeds))

    def max_episode_timesteps(self):
        return self.max_episode_length

    def reset(self):
        print("reset!")
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
        self.state = np.zeros((self.max_episode_timesteps(), len(self.all_seeds)))
        return self.state

    def execute(self, actions):
        """
        No reward until the end of the episode.
        One point for each seed measured over the whole episode.
        Maximum score is len(self.all_seeds)
        """
        seed_index = actions
        seed_measurement = self.all_seeds[seed_index](size=1)
        self.state[self.turn, seed_index] = seed_measurement
        next_state = self.state

        if self.turn < (self.max_episode_timesteps() - 1):
            terminal = False
            reward = 0.0
        else:
            terminal = True
            measurement_counts, measured_seed_count = count_measurements(
                time_steps_by_seeds_state=self.state
            )

            measured_bad_seed_count = np.sum(measurement_counts[0, self.bad_seed_indices])

            reward = 1.0 * measured_bad_seed_count

        self.turn += 1

        return next_state, terminal, reward


if __name__ == "__main__":
    print("nothing to see here")
