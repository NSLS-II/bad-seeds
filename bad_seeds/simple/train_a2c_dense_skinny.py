"""Copy agent 4 to work on ElectricBoogaloo"""
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from bad_seeds.simple.bad_seeds_skinny import BadSeedsSkinny


def tensorflow_settings():
    import tensorflow as tf
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the last GPU, and dynamically grow memory use
        try:
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def set_up():
    tensorflow_settings()
    bad_seeds_environment = Environment.create(
        environment=BadSeedsSkinny, seed_count=10, bad_seed_count=3, history_block=2, max_episode_timesteps=100,
    )

    agent = Agent.create(
        agent="a2c",
        network=[
            dict(type='flatten'),
            dict(type='dense', size=32, activation='relu'),
            dict(type='dense', size=32, activation='relu')],
        batch_size=10000,  # changed for 04 but was this a mistake? no
        horizon=50,     # changed from 100 to 50 for agent_04
        discount=0.97,  # new for agent_04
        #exploration=0.05,  # turned off for agent_04 - turn on for 05?
        l2_regularization=0.1,
        #entropy_regularization=0.2,  # turned off for agent_03
        variable_noise=0.5,  # changed from 0.1 to 0.5 for agent_04
        environment=bad_seeds_environment,
        summarizer=dict(
            directory="training_data/a2c_dense_skinny/summaries",
            # list of labels, or 'all'
            labels="all",
            frequency=100,  # store values every 100 timesteps
        ),
    )

    return bad_seeds_environment, agent


def main():
    bad_seeds_environment, agent = set_up()
    runner = Runner(agent=agent, environment=bad_seeds_environment)
    runner.run(num_episodes=10000)
    agent.save(directory="saved_models")
    bad_seeds_environment.close()
    agent.close()

if __name__ == "__main__":
    main()
