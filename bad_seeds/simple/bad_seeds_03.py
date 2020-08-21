from functools import partial

import numpy as np
import scipy.stats
from tensorforce.environments import Environment


def count_measurements(time_steps_by_seeds_state):
    measurement_indices = np.zeros_like(time_steps_by_seeds_state)
    measurement_indices[time_steps_by_seeds_state != 0.0] = 1.0

    # how many measurements were made on each seed
    _measurement_counts = np.sum(measurement_indices, axis=0, keepdims=True)

    # how many seeds were measured over the whole episode
    _measured_seed_count = np.sum(
        np.ones_like(_measurement_counts)[_measurement_counts > 0.0]
    )

    return _measurement_counts, _measured_seed_count


class BadSeeds03(Environment):
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
        self.history_array = None
        self.history_lists = None
        self.state = None

        self.reset()

    def states(self):
        """
        For each seed:
            mean
            max confidence interval of mean
            min confidence interval of mean
            standard deviation
            max confidence interval of standard deviation
            min confidence interval of standard deviation
            count of measurements per seed
        """
        return dict(
            type="float", shape=(7, len(self.all_seeds))
        )

    def actions(self):
        return dict(type="int", num_values=len(self.all_seeds))

    def max_episode_timesteps(self):
        return self.max_episode_length

    def reset(self):
        self.bad_seeds = [
            partial(self.rng.normal, loc=0.0, scale=10.0, size=1)
            for _ in range(self.bad_seed_count)
        ]
        self.good_seeds = [
            partial(self.rng.normal, loc=0.0, scale=1.0, size=1)
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

        self.history_array = np.zeros((self.max_episode_length, len(self.all_seeds)))

        self.history_lists = [list() for _ in self.all_seeds]

        self.state = np.zeros((7, len(self.all_seeds)))

        # take three measurements of all seeds
        for time_i in range(3):
            for seed_i in range(len(self.all_seeds)):
                try:
                    self.measure_seed_at_time(seed_index=seed_i, time_index=time_i)
                except ValueError:
                    # this happens the first time because
                    # there is not enough data to calculate statistics
                    pass

        # we have taken 3 turns
        self.turn = 3

        return self.state

    def measure_seed_at_time(self, seed_index, time_index):
        seed_measurement_at_time = self.all_seeds[seed_index]()
        self.history_array[time_index, seed_index] = seed_measurement_at_time

        seed_measurement_list = self.history_lists[seed_index]
        seed_measurement_list.append(seed_measurement_at_time)

        mean_seed_i, _, std_seed_i = scipy.stats.bayes_mvs(seed_measurement_list)
        self.state[0, seed_index] = mean_seed_i.statistic
        self.state[1, seed_index] = mean_seed_i.minmax[0]
        self.state[2, seed_index] = mean_seed_i.minmax[1]
        self.state[3, seed_index] = std_seed_i.statistic
        self.state[4, seed_index] = std_seed_i.minmax[0]
        self.state[5, seed_index] = std_seed_i.minmax[1]

        self.state[6, seed_index] = len(seed_measurement_list)

    def execute(self, actions):
        """
        No reward until the end of the episode.
        No reward unless all seeds are measured.
        Reward is the number of measurements made on the least-measured bad seed.
        Maximum score is approximately seed_count / 3.
        """
        seed_index = actions
        self.measure_seed_at_time(seed_index=seed_index, time_index=self.turn)
        next_state = self.state

        if self.turn < (self.max_episode_timesteps() - 1):
            terminal = False
            reward = 0.0
        else:
            terminal = True
            measurement_counts, measured_seed_count = count_measurements(
                time_steps_by_seeds_state=self.history_array
            )
            # were all seeds measured?
            if np.sum(measured_seed_count) < len(self.all_seeds):
                reward = 0.0
            else:
                bad_seed_measured_counts = measurement_counts[0, self.bad_seed_indices]
                #print(f"bad_seed_measured_counts:\n{bad_seed_measured_counts}")
                least_measured_bad_seed_count = np.min(bad_seed_measured_counts)
                #print(f"least_measured_bad_seed_count: {least_measured_bad_seed_count}")
                reward = least_measured_bad_seed_count

        self.turn += 1

        return next_state, terminal, reward


if __name__ == "__main__":
    print("nothing to see here")
