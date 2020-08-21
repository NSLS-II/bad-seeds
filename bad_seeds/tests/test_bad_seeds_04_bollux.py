from itertools import cycle

from toolz.itertoolz import concatv, take
import numpy as np
import pytest
from tensorforce.environments import Environment

from bad_seeds.simple.bad_seeds_04_bollux import Bollux, count_measurements


def test_initialization():
    bollux_env = Environment.create(
        environment=Bollux,
        seed_count=10,
        bad_seed_count=3,
        max_episode_length=100,
    )

    assert bollux_env.states()["shape"] == (7, 10)
    assert bollux_env.actions()["num_values"] == 10

    assert bollux_env.reward_probability == 0.667
    assert bollux_env.history_array.shape == (100, 10)
    assert bollux_env.state.shape == (7, 10)
    assert len(bollux_env.bad_seeds) == 3
    assert len(bollux_env.good_seeds) == 7

    # the environment is 'primed' with 3 measurements for every seed
    per_seed_measurement_count, measurement_count = count_measurements(
        bollux_env.history_array
    )
    assert np.all(per_seed_measurement_count == 3)
    # all seeds have been measured
    assert measurement_count == 10


def test_bad_initialization():
    with pytest.raises(ValueError):
        Environment.create(
            environment=Bollux,
            seed_count=3,
            bad_seed_count=10,
            max_episode_length=100,
        )


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
        time_steps_by_seeds_history=history
    )
    assert np.all(measurement_counts == np.array([1, 3, 2, 0]))
    assert measurement_count == 3


def test_play_the_game_badly():
    """
    Set up an environment with 5 steps before termination and reward probability 1.0.
    Measure the first good seed, expect no reward.
    Measure the 1st bad seed, expect no reward.
    Measure the 2nd bad seed, expect no reward.
    Measure the 3rd bad seed, expect reward of 1.0.
    Measure the first good seed again, expect no reward, expect the episode to end.
    """
    bollux_env = Bollux(
        seed_count=5, bad_seed_count=3, max_episode_length=3 + 5, reward_probability=1.0
    )

    per_seed_measurement_count, measured_seed_count = count_measurements(
        bollux_env.history_array
    )
    assert np.all(per_seed_measurement_count == np.array([3, 3, 3, 3, 3]))
    # all seeds were measured at reset()
    assert measured_seed_count == 5

    # measure the first good seed
    a_good_seed_index = bollux_env.good_seed_indices[0]
    next_state, terminal, reward = bollux_env.execute(actions=a_good_seed_index)
    assert terminal is False
    assert reward == 0.0

    # measure all bad seeds except the last one
    for bad_seed_index in bollux_env.bad_seed_indices[:-1]:
        next_state, terminal, reward = bollux_env.execute(actions=bad_seed_index)
        assert terminal is False
        assert reward == 0.0

    # measure the last bad seed
    next_state, terminal, reward = bollux_env.execute(actions=bollux_env.bad_seed_indices[-1])
    assert terminal is False
    assert reward == 1.0

    # measure the first good seed again
    # this should terminate the episode
    a_good_seed_index = bollux_env.good_seed_indices[0]
    next_state, terminal, reward = bollux_env.execute(actions=a_good_seed_index)
    assert terminal is True
    assert reward == 0.0
