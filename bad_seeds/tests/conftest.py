import pytest


@pytest.fixture
def default_environment():
    from bad_seeds.environments.cartseed import CartSeedCountdown
    from tensorforce.environments import Environment
    env = CartSeedCountdown(seed_count=10,
                            bad_seed_count=3,
                            frozen_order=True,
                            )
    env = Environment.create(environment=env)
    return env
