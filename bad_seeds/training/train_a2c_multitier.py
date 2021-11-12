from tensorforce.agents import Agent
from tensorforce.environments import Environment

from tensorforce.execution import Runner
from bad_seeds.environments.cartseed import CartSeedMutliTier
from bad_seeds.utils.tf_utils import tensorflow_settings
from pathlib import Path


def set_up(
    time_limit=None,
    gpu_idx=0,
    batch_size=16,
    seed_count=10,
    n_states=2,
    out_path=None,
):
    tensorflow_settings(gpu_idx)
    if out_path is None:
        out_path = Path().absolute()
    else:
        out_path = Path(out_path).expanduser().absolute()
    environment = CartSeedMutliTier(
        seed_count=seed_count,
        n_states=n_states,
        scans_per_state=1,
        measurement_time=time_limit,
        minimum_score=1.0,
        rng_seed=None,
    )
    env = Environment.create(environment=environment)
    agent = Agent.create(
        agent="a2c",
        batch_size=batch_size,
        environment=env,
        summarizer=dict(
            directory=out_path
            / f"training_data/a2c_multitier/{seed_count}_{n_states}_{batch_size}_{time_limit}",
            labels=[
                "entropy",
                "kl-divergence",
                "loss",
                "episode-reward",
                "update-norm",
            ],
            frequency=1,
        ),
    )
    return env, agent


def _manual_main(
    *,
    time_limit=None,
    gpu_idx=0,
    batch_size=16,
    seed_count=10,
    n_states=2,
    out_path=None,
    num_episodes=int(3 * 10 ** 3),
):
    """Manual lop for debugging purposes"""
    env, agent = set_up(
        time_limit=time_limit,
        gpu_idx=gpu_idx,
        batch_size=batch_size,
        seed_count=seed_count,
        n_states=n_states,
        out_path=out_path,
    )

    for i in range(num_episodes):
        print(f"{i}th EPISODE {'='*80}")
        states = env.reset()
        terminal = False
        episode_reward = 0
        episode_len = 0
        while not terminal:
            actions = agent.act(states=states)
            print(f"action: {'move' if actions else 'stay'}")
            states, terminal, reward = env.execute(actions)
            print(states, terminal, reward)
            episode_reward += reward
            episode_len += 1
            agent.observe(terminal=terminal, reward=reward)
        print(f"Episode reward: {episode_reward}. Episode length {episode_len}")


def main(
    *,
    time_limit=None,
    gpu_idx=0,
    batch_size=16,
    seed_count=10,
    n_states=2,
    out_path=None,
    num_episodes=int(3 * 10 ** 3),
):
    """
    A self contained set up of the environment and run.
    Running tensorboard tensorboard --logdir=training_data/a2c_cartseed will give appropriate plots of
    runs, with agent.observe/episode-reward a key metric to track.
    Parameters
    ----------
    time_limit: int, None
        Turn time limit for episode.
        Default is number of measurements needed to complete an episode to maximal goodness, and no extra.
    gpu_idx: int
        optional index for GPU
    batch_size: int
        Batch size for training
    seed_count: int
        Total number of seeds/samples to include in environment
    n_states: int
            Number of tiers of goodness, not including 'unmeasured' as a default state
    out_path: path
        Top level dir for output of models and checkpoints
    num_episodes: int
        Number of episodes to learn over

    Returns
    -------
    None

    """
    env, agent = set_up(
        time_limit=time_limit,
        gpu_idx=gpu_idx,
        batch_size=batch_size,
        seed_count=seed_count,
        n_states=n_states,
        out_path=out_path,
    )

    runner = Runner(agent=agent, environment=env)
    runner.run(num_episodes=num_episodes)
    if out_path is None:
        out_path = Path()
    else:
        out_path = Path(out_path).expanduser()
    agent.save(directory=str(out_path / "saved_models"))
    agent.close()
    env.close()
    return agent


if __name__ == "__main__":
    agent = main(num_episodes=2000, n_states=3)
