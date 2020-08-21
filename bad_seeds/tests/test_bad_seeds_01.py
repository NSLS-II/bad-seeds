from itertools import cycle

from toolz.itertoolz import concatv, take
import numpy as np
import pytest
from tensorforce.environments import Environment

from bad_seeds.simple.bad_seeds_01 import BadSeeds01, count_measurements


def test_initialization():
    bad_seeds_01_env = Environment.create(
        environment=BadSeeds01, seed_count=10, bad_seed_count=3, max_episode_length=100
    )

    assert bad_seeds_01_env.state.shape == (100, 10)
    assert len(bad_seeds_01_env.bad_seeds) == 3
    assert len(bad_seeds_01_env.good_seeds) == 7


def test_bad_initialization():
    with pytest.raises(ValueError):
        BadSeeds01(seed_count=3, bad_seed_count=10, max_episode_length=100)


def test_count_measurements():
    state = np.array(
        [
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, -0.5, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, -0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.0],
        ]
    )

    measurement_counts, measured_seed_counts = count_measurements(
        time_steps_by_seeds_state=state
    )
    assert np.all(measurement_counts == np.array([1, 3, 2, 0]))
    assert measured_seed_counts == 3


def test_play_the_game_badly():
    bad_seeds_01_env = BadSeeds01(seed_count=5, bad_seed_count=3, max_episode_length=5)

    # measure all seeds but the last seed
    for time_i, seed_i in enumerate(range(len(bad_seeds_01_env.all_seeds) - 1)):
        next_state, terminal, reward = bad_seeds_01_env.execute(actions=seed_i)
        assert next_state[time_i, seed_i] != 0.0
        assert terminal is False
        assert reward == 0.0

        # measurement_counts looks like this
        #   time_i = 0: [1 0 0 0 0 ]
        #   time_i = 1: [1 1 0 0 0 ]
        #   ...
        #   time_i = 3: [1 1 1 1 0 ]
        measurement_counts, measured_seed_counts = count_measurements(
            bad_seeds_01_env.state
        )
        for seed_j in range(seed_i):
            assert measurement_counts[0, seed_j] == 1
        assert measured_seed_counts == (seed_i + 1)

    # measure the first seed again
    # no reward because the last seed is never measured
    next_state, terminal, reward = bad_seeds_01_env.execute(actions=0)
    assert next_state[len(bad_seeds_01_env.all_seeds) - 1, 0] != 0.0
    assert terminal is True
    assert reward == 0.0

    measurement_counts, measured_seed_counts = count_measurements(
        bad_seeds_01_env.state
    )
    assert np.all(measurement_counts == np.array([[2, 1, 1, 1, 0]]))
    assert measured_seed_counts == 4


def test_play_the_game_less_badly():
    bad_seeds_01_env = BadSeeds01(
        seed_count=5, bad_seed_count=3, max_episode_length=2 * 2 + 3 * 3 + 1
    )

    # measure the good seeds twice
    # measure the bad seeds three times
    for time_i, seed_i in enumerate(
        concatv(
            take(
                n=2 * len(bad_seeds_01_env.good_seeds),
                seq=cycle(bad_seeds_01_env.good_seed_indices),
            ),
            take(
                n=3 * len(bad_seeds_01_env.bad_seeds),
                seq=cycle(bad_seeds_01_env.bad_seed_indices),
            ),
        )
    ):
        next_state, terminal, reward = bad_seeds_01_env.execute(actions=seed_i)
        assert next_state[time_i, seed_i] != 0.0
        assert terminal is False
        assert reward == 0.0

    # measure the first good seed again
    next_state, terminal, reward = bad_seeds_01_env.execute(
        actions=bad_seeds_01_env.good_seed_indices[0]
    )
    assert next_state[-1, bad_seeds_01_env.good_seed_indices[0]] != 0.0
    assert terminal is True
    # reward is the number of times the least-measured seed was measured
    assert reward == 2.0
