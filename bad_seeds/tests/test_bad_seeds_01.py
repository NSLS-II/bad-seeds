import pytest

from bad_seeds.simple.bad_seeds_01 import BadSeeds01


def test_initialization():
    bad_seeds_01_env = BadSeeds01(
        seed_count=10,
        bad_seed_count=3,
        max_episode_timesteps=100
    )

    assert bad_seeds_01_env.state.shape == (100, 10)
    assert len(bad_seeds_01_env.bad_seeds) == 3
    assert len(bad_seeds_01_env.good_seeds) == 7


def test_bad_initialization():
    with pytest.raises(ValueError):
        BadSeeds01(
            seed_count=3,
            bad_seed_count=10,
            max_episode_timesteps=100
        )


def test_play_the_game():
    bad_seeds_01_env = BadSeeds01(
        seed_count=10,
        bad_seed_count=3,
        max_episode_timesteps=2
    )

    next_state, terminal, reward = bad_seeds_01_env.execute(actions=1)
    assert next_state[0, 0] == 0.0
    assert next_state[0, 1] != 0.0
