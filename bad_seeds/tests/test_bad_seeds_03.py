from itertools import cycle

from toolz.itertoolz import concatv, take
import numpy as np
import pytest
from tensorforce.environments import Environment

from bad_seeds.simple.bad_seeds_03 import BadSeeds03, count_measurements


def test_initialization():
    bad_seeds_03_env = Environment.create(
        environment=BadSeeds03, seed_count=10, bad_seed_count=3, max_episode_length=100
    )

    assert bad_seeds_03_env.history_array.shape == (100, 10)
    assert bad_seeds_03_env.state.shape == (7, 10)
    assert len(bad_seeds_03_env.bad_seeds) == 3
    assert len(bad_seeds_03_env.good_seeds) == 7

    measurement_count_per_seed, measurement_count = count_measurements(
        bad_seeds_03_env.history_array
    )
    assert np.all(measurement_count_per_seed == 3 * np.ones((1, 10)))
    # all seeds have been measured
    assert measurement_count == 10


def test_bad_initialization():
    with pytest.raises(ValueError):
        BadSeeds03(seed_count=3, bad_seed_count=10, max_episode_length=100)


def test_count_measurements():
    history = np.array(
        [
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, -0.5, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, -0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.0],
        ]
    )

    measurement_counts, measurement_count = count_measurements(
        time_steps_by_seeds_state=history
    )
    assert np.all(measurement_counts == np.array([1, 3, 2, 0]))
    assert measurement_count == 3


def test_play_the_game_badly():
    bad_seeds_03_env = BadSeeds03(
        seed_count=5, bad_seed_count=3, max_episode_length=3 + 5
    )

    measurement_counts, measured_seed_count = count_measurements(
        bad_seeds_03_env.history_array
    )
    assert np.all(measurement_counts == np.array([3, 3, 3, 3, 3]))
    # all seeds were measured at reset()
    assert measured_seed_count == 5

    # print(f"history before start:\n{bad_seeds_03_env.history}")
    # measure all seeds but the last seed
    for time_i, seed_i in enumerate(range(len(bad_seeds_03_env.all_seeds) - 1)):
        time_i += 3

        # print(f"time_i: {time_i}")
        # print(f"turn before execute: {bad_seeds_03_env.turn}")
        next_state, terminal, reward = bad_seeds_03_env.execute(actions=seed_i)
        # print(f"turn after execute: {bad_seeds_03_env.turn}")
        # print(f"history:\n{bad_seeds_03_env.history}")
        assert bad_seeds_03_env.history_array[time_i, seed_i] != 0.0
        assert terminal is False
        assert reward == 0.0

        # measurement_counts looks like this
        #   time_i = 0: [4 3 3 3 3 ]
        #   time_i = 1: [4 4 3 3 3 ]
        #   ...
        #   time_i = 3: [4 4 4 4 3 ]
        measurement_counts, measured_seed_counts = count_measurements(
            bad_seeds_03_env.history_array
        )
        for seed_j in range(seed_i):
            # print(seed_j)
            # print(measurement_counts)
            assert measurement_counts[0, seed_j] == 4
        assert measured_seed_counts == len(bad_seeds_03_env.all_seeds)

    # measure the first seed again
    # no reward because the last seed is never measured
    next_state, terminal, reward = bad_seeds_03_env.execute(actions=4)

    # print(f"bad_seed_measured_counts: {bad_seed_measured_counts}")
    # print(f"least_measured_bad_seed_count: {least_measured_bad_seed_count}")

    assert next_state[len(bad_seeds_03_env.all_seeds) - 1, 0] != 0.0
    assert terminal is True
    assert reward == 4.0

    measurement_counts, measured_seed_counts = count_measurements(
        bad_seeds_03_env.state
    )
    assert np.all(measurement_counts == np.array([[7, 7, 7, 7, 7]]))
    assert measured_seed_counts == 5


def test_play_the_game_less_badly():
    bad_seeds_03_env = BadSeeds03(
        seed_count=5, bad_seed_count=3, max_episode_length=3 + 2 * 2 + 3 * 3 + 1
    )

    # measure the good seeds twice
    # measure the bad seeds three times
    for time_i, seed_i in enumerate(
        concatv(
            take(
                n=2 * len(bad_seeds_03_env.good_seeds),
                seq=cycle(bad_seeds_03_env.good_seed_indices),
            ),
            take(
                n=3 * len(bad_seeds_03_env.bad_seeds),
                seq=cycle(bad_seeds_03_env.bad_seed_indices),
            ),
        )
    ):
        time_i += 3
        next_state, terminal, reward = bad_seeds_03_env.execute(actions=seed_i)
        assert bad_seeds_03_env.history_array[time_i, seed_i] != 0.0
        assert terminal is False
        assert reward == 0.0

    measurement_counts, measured_seed_counts = count_measurements(
        bad_seeds_03_env.history_array
    )
    expected_measurement_counts = np.zeros_like(measurement_counts)
    expected_measurement_counts[0, bad_seeds_03_env.good_seed_indices] = 5
    expected_measurement_counts[0, bad_seeds_03_env.bad_seed_indices] = 6
    assert np.all(measurement_counts == expected_measurement_counts)

    # measure the first good seed again
    next_state, terminal, reward = bad_seeds_03_env.execute(
        actions=bad_seeds_03_env.good_seed_indices[0]
    )

    print(f"history:\n{bad_seeds_03_env.history_array}")
    measurement_counts, measured_seed_counts = count_measurements(
        bad_seeds_03_env.history_array
    )
    print(f"measurement_counts: {measurement_counts}")

    assert next_state[-1, bad_seeds_03_env.good_seed_indices[0]] != 0.0
    assert terminal is True
    # reward is the number of times the least-measured seed was measured
    assert reward == 6.0

    expected_measurement_counts[0, bad_seeds_03_env.good_seed_indices[0]] += 1
    assert np.all(measurement_counts == expected_measurement_counts)
