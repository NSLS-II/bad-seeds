from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from bad_seeds.simple.bad_seeds_cart import CartSeed01


def tensorflow_settings():
    import tensorflow as tf
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the last GPU, and dynamically grow memory use
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def set_up():
    tensorflow_settings()
    env = Environment.create(
        environment=CartSeed01,
        seed_count=10,
        bad_seed_count=None,
        max_count=10,
        sequential=True
    )

    agent = Agent.create(
        agent="a2c",
        batch_size=1,
        # Best and slowest batch size is 1, larger batch size will decrease speed
        # The following were shown effective in prelim tests and should be optimized with
        # and appropriate hyperparameter opt scheme: discount, exploration, l2_regularization.
        # discount=0.97,
        # exploration=0.05,
        # l2_regularization=0.1,
        environment=env,
        summarizer=dict(
            directory="training_data/a2c_cartseed/summaries",
            labels="all",
            frequency=1,
        ),
        # saver=dict(
        #     directory='saved_models/agent_04_env_04_1000/checkpoints',
        #     frequency=600  # save checkpoint every 600 seconds (10 minutes)
        # ),
    )
    return env, agent


def main():
    env, agent = set_up()
    runner = Runner(agent=agent, environment=env)
    runner.run(num_episodes=int(3*10**3))
    agent.save(directory="saved_models")
    agent.close()
    env.close()


def manual_main():
    env, agent = set_up()
    for i in range(100):
        states = env.reset()
        terminal = False
        episode_reward = 0
        episode_len = 0
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = env.execute(actions=actions)
            episode_reward += reward
            episode_len += 1
            agent.observe(terminal=terminal, reward=reward)
        print(f"Episode reward: {episode_reward}. Episode length {episode_len}")


if __name__ == "__main__":
    main()
