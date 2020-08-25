from itertools import cycle

from toolz.itertoolz import concatv, take
import numpy as np
import pytest
from tensorforce.environments import Environment

from bad_seeds.simple.bad_seeds_skinny import BadSeedsSkinny


def test_initialization():
    env = Environment.create(
        environment=BadSeedsSkinny,
        seed_count=10,
        bad_seed_count=3,
        history_block=2,
        max_episode_timesteps=100,
    )

    assert env.states()["shape"] == (10, 2)
    assert env.actions()["num_values"] == 10
    env.reset()
    assert len(env.environment.history) == 0
    assert len(env.environment.bad_seeds) == 3
    assert len(env.environment.good_seeds) == 7
    # Full exp is the equivalent of history array
    assert np.all([len(a) == 2 for a in env.full_exp])


def test_bad_initialization():
    with pytest.raises(ValueError):
        env = Environment.create(environment=BadSeedsSkinny,
                                 seed_count=3,
                                 bad_seed_count=10,
                                 max_episode_timesteps=100)


def test_play_the_game_badly():
    """
    Set up environement and run for set of turns.
    1. Good seed, no reward
    2. Bad seed, reward
    3. Bad seed, reward
    4. Repeat 2, penalty
    5. Repeat 4, penalty
    6. Repeat 1, no reward
    7. Repeat 1, penalty
    8. Repeat 2, reward and terminal

    Returns
    -------

    """
    env = Environment.create(
        environment=BadSeedsSkinny, seed_count=5, bad_seed_count=3, max_episode_timesteps=8, history_block=2, reward_ratio=0.
    )
    env.reset()

    assert np.all([len(a) == 2 for a in env.full_exp])
    # all seeds were measured at reset()

    # measure the first good seed
    a_good_seed_index = env.good_seed_indices[0]
    next_state, terminal, reward = env.execute(actions=a_good_seed_index)
    assert bool(terminal) is False
    assert reward == 0.0

    # measure all bad seeds except the last one
    bad_seed_0 = env.bad_seed_indices[0]
    bad_seed_1 = env.bad_seed_indices[1]
    next_state, terminal, reward = env.execute(actions=bad_seed_0)
    assert bool(terminal) is False
    assert reward == 1.0
    next_state, terminal, reward = env.execute(actions=bad_seed_1)
    assert bool(terminal) is False
    assert reward == 1.0

    # Measure repeats
    next_state, terminal, reward = env.execute(actions=bad_seed_0)
    assert bool(terminal) is False
    assert reward == -1.0
    next_state, terminal, reward = env.execute(actions=bad_seed_0)
    assert bool(terminal) is False
    assert reward == -1.0

    # Good Seed Repeats
    next_state, terminal, reward = env.execute(actions=a_good_seed_index)
    assert bool(terminal) is False
    assert reward == 0.0
    next_state, terminal, reward = env.execute(actions=a_good_seed_index)
    assert bool(terminal) is False
    assert reward == -1.0

    #Terminal
    next_state, terminal, reward = env.execute(actions=bad_seed_0)
    assert bool(terminal) is True
    assert reward == 1.0
