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
    bad_seeds_01_env = BadSeeds01(seed_count=5, bad_seed_count=3, max_episode_length=2)

    a_good_seed_ndx = bad_seeds_01_env.good_seed_indices[0]

    next_state, terminal, reward = bad_seeds_01_env.execute(actions=a_good_seed_ndx)
    assert next_state[0, a_good_seed_ndx] != 0.0
    assert terminal is False
    assert reward == 0.0

    measurement_counts, measured_seed_counts = count_measurements(
        bad_seeds_01_env.state
    )
    print(measurement_counts)
    expected_measurement_counts = np.zeros_like(measurement_counts)
    expected_measurement_counts[0, a_good_seed_ndx] += 1.0
    assert np.all(measurement_counts == expected_measurement_counts)
    assert measured_seed_counts == 1

    next_state, terminal, reward = bad_seeds_01_env.execute(actions=a_good_seed_ndx)
    assert next_state[1, a_good_seed_ndx] != 0.0
    assert terminal is True
    assert reward == 0.0

    measurement_counts, measured_seed_counts = count_measurements(
        bad_seeds_01_env.state
    )
    print(measurement_counts)
    expected_measurement_counts[0, a_good_seed_ndx] += 1.0
    assert np.all(measurement_counts == expected_measurement_counts)
    assert measured_seed_counts == 1


def test_play_the_game_less_badly():
    bad_seeds_01_env = BadSeeds01(seed_count=5, bad_seed_count=3, max_episode_length=2)

    a_bad_seed_ndx = bad_seeds_01_env.bad_seed_indices[0]

    next_state, terminal, reward = bad_seeds_01_env.execute(actions=a_bad_seed_ndx)
    assert next_state[0, a_bad_seed_ndx] != 0.0
    assert terminal is False
    assert reward == 0.0

    measurement_counts, measured_seed_counts = count_measurements(
        bad_seeds_01_env.state
    )
    print(measurement_counts)
    expected_measurement_counts = np.zeros_like(measurement_counts)
    expected_measurement_counts[0, a_bad_seed_ndx] += 1.0
    assert np.all(measurement_counts == expected_measurement_counts)
    assert measured_seed_counts == 1

    next_state, terminal, reward = bad_seeds_01_env.execute(actions=a_bad_seed_ndx)
    assert next_state[1, a_bad_seed_ndx] != 0.0
    assert terminal is True
    assert reward == 2.0

    measurement_counts, measured_seed_counts = count_measurements(
        bad_seeds_01_env.state
    )
    print(measurement_counts)
    expected_measurement_counts[0, a_bad_seed_ndx] += 1.0
    assert np.all(measurement_counts == expected_measurement_counts)
    assert measured_seed_counts == 1
