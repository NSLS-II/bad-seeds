from pathlib import Path

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from bad_seeds.simple.bad_seeds_02 import BadSeeds02


def tensorflow_settings():
    import tensorflow as tf
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the last GPU, and dynamically grow memory use
        try:
            tf.config.experimental.set_visible_devices(gpus[-1], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def main():
    tensorflow_settings()
    bad_seeds_environment = Environment.create(
        environment=BadSeeds02, seed_count=10, bad_seed_count=3, history_block=2, max_episode_timesteps=100,
    )

    agent = Agent.create(
        agent="random",
        environment=bad_seeds_environment,
        summarizer=dict(
            directory="training_data/agent_random_env_02/summaries",
            labels="all",
            frequency=100,  # store values every 100 timesteps
        ),
    )

    runner = Runner(agent=agent, environment=bad_seeds_environment)
    runner.run(num_episodes=10000)

    bad_seeds_environment.close()
    agent.close()


if __name__ == "__main__":
    main()
