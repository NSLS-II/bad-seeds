from collections import deque
import numpy as np
from tensorforce.environments import Environment


class CartSeed01(Environment):

    def __init__(self, seed_count, *, bad_seed_count=None, max_count=10, frozen_order=False, sequential=False,
                 revisiting=True):
        """
        Bad seeds, but make it cartpole...

        Assuming the envrionment experiences two kinds of seeds:
            - Good Seeds that no longer need to be sampled
            - Bad Seeds that need to be sampled a fixed amount

        This allows for a deterministic high score that a well behaved agent will approach.
        The key assumptions of this framing are that from an initial sampling of all seeds (brief scans of all samples)
        it will be clear which are Bad and which are Good. This should be extensible to varying degrees of goodness.

        Parameters
        ----------
        seed_count: int
            Number of total seeds
        bad_seed_count: int
            Number of bad seeds. If None, a variable amount will be used for each reset.
        max_count: int
            Maximum number of samples/scans needed to saturate a bad_seed
        frozen_order: bool
            For debugging or an easier game. This locks the order of the seeds and order of the sampling.
            Bad seeds are the first set of seeds.
        sequential: bool
            Visit the samples in sequential order, not randomly.
        revisiting: bool
            Whether to allow revisiting of past samples. Once all samples are visited, the memory resets.
            The memory is a hashable set that gets emptied when its length reaches the seed count.
            A possible update is to make this a terminal condition.
        """
        super().__init__()

        if bad_seed_count is None:
            self.variable_bad_seed = True
            self.bad_seed_count = 0
        elif bad_seed_count > seed_count:
            raise ValueError("bad_seed_count must be less than or equal to seed_count")
        else:
            self.bad_seed_count = bad_seed_count
            self.variable_bad_seed = False
        self.seed_count = seed_count
        self.bad_seed_reward = 1
        self.good_seed_reward = 0
        self.max_count = max_count
        self.frozen_order = bool(frozen_order)
        self.sequential_order = bool(sequential)
        self.revisiting = bool(revisiting)
        self.visited = set()
        self.timestep = 0

        self.seeds = np.empty((seed_count, 2))
        self.current_idx = None
        self.exp_sequence = []

        self.bad_seed_indicies = None
        self.good_seed_indicies = None

        self.rng = np.random.default_rng()

    def states(self):
        """
        State is current seed [bool(bad), countdown]

        Returns
        -------
        state specification
        """
        return dict(type="float", shape=(2,))

    def actions(self):
        """
        Actions specification: Stay or go
        Returns
        -------
        Action spec
        """
        return dict(type="int", num_values=2)

    def max_episode_timesteps(self):
        """
        Returns
        -------
        Maximum count equivalent to maximum possible score plus required moves to get there
        """
        return self.max_count * self.bad_seed_count + self.seed_count

    def reset(self):
        """
        Sets up seeds array and indicies. Plenty of redundant tracking.
        If frozen order is set, then the first 3 indicies are always bad seeds.
        If variable bad seed, the bad seed count is randomly varied, and the max score is kept at 100.
        Returns
        -------
        State
        """
        self.timestep = 0
        l = list(range(self.seed_count))
        if not self.frozen_order:
            self.rng.shuffle(l)
        if self.variable_bad_seed:
            self.bad_seed_count = self.rng.integers(self.seed_count)
            if self.bad_seed_count > 0:
                self.bad_seed_reward = 100 / (self.bad_seed_count*self.max_count)
            else:
                self.bad_seed_reward = 1
        self.bad_seed_indicies = l[:self.bad_seed_count]
        self.good_seed_indicies = l[self.bad_seed_count:]
        self.seeds[self.bad_seed_indicies, :] = [1, self.max_count]
        self.seeds[self.good_seed_indicies, :] = [0, 0]

        self.current_idx = self.rng.integers(self.seed_count)
        self.exp_sequence.append(self.current_idx)
        state = self.seeds[self.current_idx, :]
        return state

    def execute(self, actions):
        """
        Updates timestep
        Updates state if moved
        Updates overall seed tracking (countdown)
        Calculates reward based on current seed and positive countdown

        Parameters
        ----------
        action: bool

        Returns
        -------
        next_state: array
        terminal: bool
        reward: float
        """
        self.timestep += 1
        move = bool(actions)
        prev_index = self.current_idx
        if move:
            # Clear previously visited or complete episode if all samples visited
            if len(self.visited) == self.seed_count:
                if not self.revisiting:
                    state = self.seeds[prev_index, :]
                    terminal = True
                    reward = self.good_seed_reward
                    return state, terminal, reward
                else:
                    self.visited = set()
            # Frozen order  and sequential order iterates
            if self.frozen_order or self.sequential_order:
                self.current_idx = (self.current_idx + 1) % self.seed_count
            # Otherwise random change that hasn't been visited
            else:
                self.current_idx = self.rng.integers(self.seed_count)
                while self.current_idx in self.visited or self.current_idx == prev_index:
                    self.current_idx = self.rng.integers(self.seed_count)
        # Add to memory
        if not self.revisiting:
            self.visited.add(self.current_idx)

        self.exp_sequence.append(self.current_idx)
        state = self.seeds[self.current_idx, :]

        if bool(self.seeds[self.current_idx, 0]) and self.seeds[self.current_idx, 1] > 0:
            reward = self.bad_seed_reward
        else:
            reward = self.good_seed_reward

        self.seeds[self.current_idx, 1] -= 1

        if self.timestep >= self.max_episode_timesteps():
            terminal = True
        else:
            terminal = False

        return state, terminal, reward


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    env = Environment.create(environment=CartSeed01, seed_count=3, bad_seed_count=1, sequential=True, revisiting=False)
    state = env.reset()
    print(f'Start state: {state}')
    print(f"Environmental snaphot:\n {env.seeds}")
    print(f"Number of bad seeds: {env.bad_seed_count}")
    for _ in range(4):
        a = True
        s, t, r = env.execute(a)
        print(f'New seed state: {s}. New seed reward: {r}. Terminal: {t}')
    print(f"Max timesteps {env.max_episode_timesteps()}")
