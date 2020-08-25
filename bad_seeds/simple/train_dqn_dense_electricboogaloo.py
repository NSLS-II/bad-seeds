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


def set_up():
    tensorflow_settings()
    bad_seeds_environment = Environment.create(
        environment=BadSeeds02, seed_count=10, bad_seed_count=3, history_block=2, max_episode_timesteps=500,
    )

    agent = Agent.create(
        agent="dqn",
        network=[
            dict(type='flatten'),
            dict(type='dense', size=32, activation='tanh'),
            dict(type='dense', size=32, activation='tanh')],
        environment=bad_seeds_environment,
        batch_size=256,
        memory=int(10 ** 7),
        exploration=0.15,
        summarizer=dict(
            directory="training_data/agent_02_env_02/summaries",
            labels="all",
            frequency=100  # store values every 100 timesteps
        )
    )

    return bad_seeds_environment, agent


def main():
    bad_seeds_environment, agent = set_up()
    runner = Runner(agent=agent, environment=bad_seeds_environment)
    runner.run(num_episodes=1000)
    agent.save(directory="saved_models")


def manual_main():
    bad_seeds_environment, agent = set_up()

    # Train 10 steps
    for i in range(10):
        states = bad_seeds_environment.reset()
        terminal = False
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = bad_seeds_environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)


if __name__ == "__main__":
    main()
