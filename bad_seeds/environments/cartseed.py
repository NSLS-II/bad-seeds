import numpy as np
from tensorforce.environments import Environment


class CartSeed(Environment):
    def __init__(
        self,
        seed_count,
        *,
        bad_seed_count=None,
        max_count=10,
        frozen_order=False,
        sequential=False,
        revisiting=True,
        bad_seed_reward_f=None,
        good_seed_reward_f=None,
        measurement_time=None,
    ):
        """
        Bad seeds, but make it cartpole...

        Assuming the envrionment experiences two kinds of seeds:
            - Good Seeds that no longer need to be sampled
            - Bad Seeds that need to be sampled a fixed amount

        This allows for a deterministic high score that a well behaved agent will approach.
        The key assumptions of this framing are that from an initial sampling of all seeds (brief scans of all samples)
        it will be clear which are Bad and which are Good. This should be extensible to varying degrees of goodness.

        Scores are default scaled to 100.
        If a bad_seed_reward_f is given, no scaling is done unelss a point max is given.
        Parameters
        ----------
        seed_count: int
            Number of total seeds
        bad_seed_count: int, None
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
        bad_seed_reward: function
            Function of the form f(state, terminal, action). Where the state is the resultant state from the action.
        good_seed_reward: function
            Function of the form f(state, terminal, action). Where the state is the resultant state from the action.
        measurement_time: int, None
            Override for max_episode_timesteps in Environment.create().
            Passing a value of max_episode_timesteps to Environment.create() will override measurement_time and the
            default max_episode_timesteps(), raising an UnexpectedError if the override value is greater than the others.
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
        # Hidden functions that get rescaled
        if bad_seed_reward_f is None:
            self._bad_seed_reward_f = lambda s, t, a: 1
        else:
            self._bad_seed_reward_f = bad_seed_reward_f
        self.bad_seed_reward_f = None  # Get's set and rescaled on reset()
        if good_seed_reward_f is None:
            self.good_seed_reward_f = lambda s, t, a: 0
        else:
            self.good_seed_reward_f = good_seed_reward_f

        self.max_count = max_count
        self.frozen_order = bool(frozen_order)
        self.sequential_order = bool(sequential)
        self.revisiting = bool(revisiting)
        self.measurement_time = measurement_time
        self.visited = set()
        self.timestep = 0

        self.seeds = np.empty((seed_count, 2))
        self.current_idx = None
        self.exp_sequence = []

        self.bad_seed_indicies = None
        self.good_seed_indicies = None

        self.rng = np.random.default_rng()

    def bad_seed_reward(self, state, terminal, action):
        """
        Functional approach to the bad seed reward
        Parameters
        ----------
        state: array
            Current state
        terminal: bool
            Current terminal status
        action: array
            Action preceeding the current state
        Returns
        -------
        reward
        """
        return self.bad_seed_reward_f(state, terminal, action)

    def good_seed_reward(self, state, terminal, action):
        """
        Functional approach to the good seed reward
        Parameters
        ----------
        state: array
            Current state
        terminal: bool
            Current terminal status
        action: array
            Action preceeding the current state
        Returns
        -------
        reward
        """
        return self.good_seed_reward_f(state, terminal, action)

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
        Maximum count equivalent to maximum possible score plus required moves to get there.
        Is overridden by the use inclusion of max_episode_timesteps in Environment.create() kwargs.
        (This uses a hidden variable from tensorforce.Environment)
        """
        if self.measurement_time is None:
            return self.max_count * self.bad_seed_count + self.seed_count
        else:
            return self.measurement_time

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

        # Always scales the reward such that the optimal performance is 100
        # Does this for defaults as well as calculating optimal points for input functions
        if self.bad_seed_count > 0:
            point_max = (
                np.sum(
                    [
                        self._bad_seed_reward_f([1, p], None, None)
                        for p in range(self.max_count, 0, -1)
                    ]
                )
                * self.bad_seed_count
            )
            self.bad_seed_reward_f = (
                lambda s, t, a: self._bad_seed_reward_f(s, t, a) * 100 / point_max
            )
        else:
            self.bad_seed_reward_f = self._bad_seed_reward_f

        self.bad_seed_indicies = l[: self.bad_seed_count]
        self.good_seed_indicies = l[self.bad_seed_count :]
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
                    reward = self.good_seed_reward(state, terminal, actions)
                    return state, terminal, reward
                else:
                    self.visited = set()
            # Frozen order  and sequential order iterates
            if self.frozen_order or self.sequential_order:
                self.current_idx = (self.current_idx + 1) % self.seed_count
            # Otherwise random change that hasn't been visited
            else:
                self.current_idx = self.rng.integers(self.seed_count)
                while (
                    self.current_idx in self.visited or self.current_idx == prev_index
                ):
                    self.current_idx = self.rng.integers(self.seed_count)
        # Add to memory
        if not self.revisiting:
            self.visited.add(self.current_idx)

        self.exp_sequence.append(self.current_idx)
        state = self.seeds[self.current_idx, :]

        if self.timestep >= self.max_episode_timesteps():
            terminal = True
        else:
            terminal = False

        if (
            bool(self.seeds[self.current_idx, 0])
            and self.seeds[self.current_idx, 1] > 0
        ):
            reward = self.bad_seed_reward(state, terminal, actions)
        else:
            reward = self.good_seed_reward(state, terminal, actions)

        self.seeds[self.current_idx, 1] -= 1

        return state, terminal, reward


class CartSeedCountdown(CartSeed):
    """
    CartSeed01, with variable countdown (proxy for badness of seed), and no boolean in state

    Assuming the envrionment experiences two kinds of seeds:
            - Good Seeds that no longer need to be sampled
            - Bad Seeds that need to be sampled a randomly initialized amount (less than or equal to max_count)

    See CartSeed for further details which are identical.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_max_count = self.max_count * self.bad_seed_count

    def reset(self):
        super().reset()
        for i in self.bad_seed_indicies:
            self.seeds[i, 1] = self.rng.integers(1, self.max_count)

        # Always scales the reward such that the optimal performance is 100
        # Does this for defaults as well as calculating optimal points for input functions
        if self.bad_seed_count > 0:
            point_max = np.sum(
                [
                    self._bad_seed_reward_f(self.seeds[i, 1], None, None)
                    * self.seeds[i, 1]
                    for i in self.bad_seed_indicies
                ]
            )
            self.bad_seed_reward_f = (
                lambda s, t, a: self._bad_seed_reward_f(s, t, a) * 100 / point_max
            )
        else:
            self.bad_seed_reward_f = self._bad_seed_reward_f

        self.total_max_count = np.sum(self.seeds[:, 1])
        state = np.array([self.seeds[self.current_idx, 1]])
        return state

    def max_episode_timesteps(self):
        """
        Returns
        -------
        Maximum count equivalent to maximum possible score plus required moves to get there.
        Is overridden by the use inclusion of max_episode_timesteps in Environment.create() kwargs.
        (This uses a hidden variable from tensorforce.Environment)
        """
        if self.measurement_time is None:
            return self.total_max_count + self.seed_count
        else:
            return self.measurement_time

    def states(self):
        """
        State is current seed [countdown]

        Returns
        -------
        state specification
        """
        return dict(type="float", shape=(1,))

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
                    state = np.array([self.seeds[prev_index, 1]])
                    terminal = True
                    reward = self.good_seed_reward(state, terminal, actions)
                    return state, terminal, reward
                else:
                    self.visited = set()
            # Frozen order  and sequential order iterates
            if self.frozen_order or self.sequential_order:
                self.current_idx = (self.current_idx + 1) % self.seed_count
            # Otherwise random change that hasn't been visited
            else:
                self.current_idx = self.rng.integers(self.seed_count)
                while (
                    self.current_idx in self.visited or self.current_idx == prev_index
                ):
                    self.current_idx = self.rng.integers(self.seed_count)
        # Add to memory
        if not self.revisiting:
            self.visited.add(self.current_idx)

        self.exp_sequence.append(self.current_idx)
        state = np.array([self.seeds[self.current_idx, 1]])

        if self.timestep >= self.max_episode_timesteps():
            terminal = True
        else:
            terminal = False

        if state > 0:
            reward = self.bad_seed_reward(state, terminal, actions)
        else:
            reward = self.good_seed_reward(state, terminal, actions)

        self.seeds[self.current_idx, 1] -= 1

        return state, terminal, reward


class CartSeedMutliTier(Environment):
    def __init__(
        self,
        seed_count,
        *,
        n_states=2,
        scans_per_state=1,
        measurement_time=None,
        minimum_score=1.0,
        rng_seed=None,
    ):
        """
        Samples should produce maximal reward on first measurement.
        Then there will be n_tiers of goodness with proportionate rewards such that
        one measurement in on one sample in a tier produces the same reward as measuring
        all samples in the next tier.

        By this notion, measuring one 'good' sample once, is more valuable than
        measuring all of the other 'good' samples twice.

        Each seed will be tracked as [baseline quality, current quality, measurement count].

        Note: countdown as high as number of tiers * scans per state

        Parameters
        ----------
        seed_count: int
            Number of seeds (aka number of samples in measurement bracket)
        n_states: int
            Number of tiers of goodness, not including 'unmeasured' as a default state
        scans_per_state: int
            Number of scans/measurements required to transition to a better state.
        measurement_time: int, None
            Override for max_episode_timesteps in Environment.create().
            Passing a value of max_episode_timesteps to Environment.create() will override measurement_time and the
            default max_episode_timesteps(), raising an UnexpectedError if the override value is greater than the others.
        minimum_score: float
            Score assinged to baseline measurement of best quality seed/sample.
            Must be greater than zero, and can be set to a small value for the sake of scaling back
            the scores of many states and/or many seeds.
        rng_seed: None, int
            Random number generator seed
        """
        super().__init__()
        self.n_states = n_states + 1  # Add 1 for the unmeasured state
        self.measurement_time = measurement_time
        self.seed_count = seed_count
        self.minimum_score = minimum_score
        self.scans_per_state = scans_per_state

        self.timestep = 0
        self.seeds = np.empty((self.seed_count, 3))
        self.current_idx = None
        self.exp_sequence = []

        self.rng = np.random.default_rng(rng_seed)

    def states(self):
        """
        State is current sample tier.
        [0 - not measured
         1 - bad
         2 - better
         3 - more better
         ...
         n - good
        ]
        Returns
        -------
        state specification

        """
        return dict(type="int", num_values=self.n_states, shape=1)

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
        max_episode_timesteps: int
            Is overridden by the use inclusion of max_episode_timesteps in Environment.create() kwargs.
            (This uses a hidden variable from tensorforce.Environment)
        """
        if self.measurement_time is None:
            return np.sum(self.seeds[:, 0] * self.scans_per_state + self.seed_count)
        else:
            return self.measurement_time

    def reset(self):
        """
        Set up seeds array and indicies.
        Each seed is randomly assigned
        Moves current index back to start

        Returns
        -------
        state
        """
        self.seeds[:, 0] = self.rng.integers(1, self.n_states, size=self.seed_count)
        self.seeds[:, 1] = 0
        self.seeds[:, 2] = 0.0
        self.current_idx = 0

        state = self.seeds[self.current_idx, 1:2]  # slicing for shape array not scalar
        return state

    def _progress_state(self, idx):
        """
        Progresses a seed state by updating its count and current state
        3 member seed [baseline, current, count]
        Parameters
        ----------
        idx: int

        Returns
        -------

        """
        # This works because every count starts at 0, and every current starts at 0
        self.seeds[idx, 2] += 1
        self.seeds[idx, 1] = min(
            self.n_states - 1,
            self.seeds[idx, 0]
            + np.floor((self.seeds[idx, 2] - 1) / self.scans_per_state),
        )

    def execute(self, action):
        """
        Updates timestep
        Updates state
        Movement here continually cycles seed, and does away with experimental movement of
        CartSeed and CartSeedCountdown
        Updates overall seed tracking (countdown)
        Calculates reward based on current seed

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
        move = bool(action)
        if move:
            self.current_idx = (self.current_idx + 1) % self.seed_count

        self.exp_sequence.append(self.current_idx)
        # Calculate reward
        state = self.seeds[self.current_idx, 1]
        reward = self.minimum_score * self.seed_count ** (self.n_states - state - 1)
        # Update state and environment
        self._progress_state(self.current_idx)
        state = self.seeds[self.current_idx, 1:2]  # Slicing for shaped array not scalar

        if self.timestep >= self.max_episode_timesteps():
            terminal = True
        else:
            terminal = False

        return state, terminal, reward


if __name__ == "__main__":
    np.set_printoptions(precision=3)

    environment = CartSeedMutliTier(
        seed_count=3, n_states=3, scans_per_state=2, measurement_time=None, rng_seed=12
    )
    env = Environment.create(environment=environment)
    state = env.reset()
    print(f"Start state: {state}")
    print(f"Environmental snaphot:\n {env.seeds}")
    print(f"Max timesteps {env.max_episode_timesteps()}")
    for a in [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
    ]:
        s, t, r = env.execute(a)
        print(
            f"New seed: {env.current_idx} State: {s}. New seed reward: {r}. Terminal: {t}"
        )
        print(f"Snapshot: {env.seeds}")
