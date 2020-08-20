from functools import partial
from collections import deque

import numpy as np
from tensorforce.environments import Environment


class BadSeeds02(Environment):
    def __init__(self, seed_count, bad_seed_count, history_block=0, reward_ratio=0.1,
                 bad_mean=0., bad_var=10., good_mean=0., good_var=1.):
        """
        Bad Seeds 2: Electric Boogaloo

        Borrowing a bit from BadSeeds01, the concept here is not to hold onto the entire history of a measurement.
        Intially every arm gets pulled 3 times.

        Each arm (or seed) is a randomly shuffled generator with a good/bad variance according to bad_seed_count.
        A state retains the sample mean, and sample variance for each seed.
        The state also retains a history to block repeated measurements.
        This assumes the cost of choosing a new sample is low, and repeated measurements are inherently wasteful

        An action can only pull 1 arm.

        The reward is set to 1 if a bad seed is pulled, and 0.1 if a good seed is pulled.

        Parameters
        ----------
        seed_count: int
            Number of total seeds
        bad_seed_count: int
            Number of bad seeds
        history_block: int
            Memory of trials to ignore
        reward_ratio: float
            [0, 1), ratio of score for pulling a good seed to pulling a bad seed
        bad_mean: float
        bad_var: float
        good_mean: float
        good_var: float

        """
        super().__init__()
        if bad_seed_count > seed_count:
            raise ValueError("bad_seed_count must be less than or equal to seed_count")
        if history_block >= seed_count:
            raise ValueError("history_block must be less than seed_count")
        self.seed_count = seed_count
        self.bad_seed_count = bad_seed_count
        self.history = deque(maxlen=history_block)
        self.bad_seed_reward = 1
        self.good_seed_reward = self.bad_seed_reward * reward_ratio
        self.bad_mean = bad_mean
        self.bad_var = bad_var
        self.good_mean = good_mean
        self.good_var = good_var

        self.rng = np.random.default_rng()

        self.timestep = 0
        self.bad_seeds = None
        self.good_seeds = None
        self.all_seeds = None
        self.bad_seed_indices = None
        self.good_seed_indices = None
        self.state = None
        self.state_shape = (self.seed_count, 3)
        # self.reset()

    def states(self):
        """
        State is [[mean, std, count]]
        Returns
        -------
        state specification
        """
        return dict(type="float", shape=self.state_shape)

    def actions(self):
        """
        Action specification: Choosing a seed/arm to measure/pull
        Returns
        -------
        Action specification
        """
        return dict(type="int", num_values=self.seed_count)

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def close(self):
        super().close()

    def reset(self):
        """
        Sets up generators for bad and good seeds, and shuffles them into all seeds.
        Initializes state and timesteps. Pulls every arm twice.
        Returns
        -------

        """
        self.bad_seeds = [
            partial(self.rng.normal, loc=self.bad_mean, scale=self.bad_var)
            for _ in range(self.bad_seed_count)
        ]
        self.good_seeds = [
            partial(self.rng.normal, loc=self.good_mean, scale=self.good_var)
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

        self.timestep = 0

        self.state = np.zeros(self.state_shape)
        for i, seed in enumerate(self.all_seeds):
            measurements = seed(size=2)
            self.state[i, :] = [np.mean(measurements), np.var(measurements), 2]

        return self.state

    @staticmethod
    def update_mean(prev_mean, prev_count, new_measurement):
        return (prev_mean*prev_count+new_measurement)/(prev_count+1)

    @staticmethod
    def update_var(prev_var, prev_mean, prev_count, new_measurement):
        return prev_count/(prev_count+1) * (prev_var + (new_measurement-prev_mean)**2/(prev_count+1))

    def execute(self, actions):
        """
        Updates timestep
        Updates state with new mean, variance, and count
        Calculates reward based on choice of seed

        Parameters
        ----------
        actions: int
            Seed index

        Returns
        -------
        next_state: array
        termnial: bool
        reward: float
        """
        self.timestep += 1
        mean_index, var_index, count_index = 0, 1, 2
        seed_index = actions
        seed_measurement = self.all_seeds[seed_index]()
        prev_count = self.state[seed_index, count_index]
        prev_mean = self.state[seed_index, mean_index]
        self.state[seed_index, mean_index] = self.update_mean(prev_mean, prev_count, seed_measurement)
        self.state[seed_index, var_index] = self.update_var(self.state[seed_index, var_index],
                                                            prev_mean,
                                                            self.state[seed_index, mean_index],
                                                            seed_measurement)
        self.state[seed_index, count_index] += 1

        if self.timestep >= self.max_episode_timesteps():
            terminal = True
        else:
            terminal = False

        if seed_index in self.history:
            reward = -1*self.bad_seed_reward
        elif seed_index in self.bad_seed_indices:
            reward = self.bad_seed_reward
        elif seed_index in self.good_seed_indices:
            reward = self.good_seed_reward
        else:
            raise NotImplementedError("Seed index neither in good nor bad seed list.")

        self.history.append(seed_index)
        return self.state, terminal, reward



if __name__ == "__main__":
    env = Environment.create(environment=BadSeeds02, seed_count=10, bad_seed_count=3, history_block=1,
                             max_episode_timesteps=3)
    env.reset()
    print(env.state)
    a = env.environment.bad_seed_indices[0]
    s, t, r = env.execute(a)
    print(f'Bad seed reward: {r}. Terminal: {t}')
    a = env.environment.good_seed_indices[0]
    s, t, r = env.execute(a)
    print(f'Good seed reward: {r}. Terminal: {t}')
    s, t, r = env.execute(a)
    print(f'Repeat seed reward: {r}. Terminal: {t}')
