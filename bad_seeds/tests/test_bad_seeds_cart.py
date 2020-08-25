import numpy as np
import pytest
from tensorforce.environments import Environment

from bad_seeds.simple.bad_seeds_cart import CartSeed01

#Initialization
#Reset
#bad initialization
#play
def test_initialization():
    env = Environment.create(
        environment=CartSeed01,
        seed_count=10,
        bad_seed_count=3,
        max_count=20
    )