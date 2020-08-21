from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from bad_seeds.simple.bad_seeds_03 import BadSeeds03


def main():

    bad_seeds_environment = Environment.create(
        environment=BadSeeds03, seed_count=10, bad_seed_count=3, max_episode_length=100
    )

    agent = Agent.create(
        agent="a2c",
        batch_size=100,
        horizon=100,     # changed from 20 to 100 for agent_03
        exploration=0.05,  # changed from 0.01 to 0.05 for agent_03
        l2_regularization=0.2,  # changed from 0.1 to 0.2 for agent_03
        #entropy_regularization=0.2,  # turned off for agent_03
        variable_noise=0.1,  # changed from 0.05 to 0.1 for agent_03
        environment=bad_seeds_environment,
        summarizer=dict(
            directory="training_data/agent_03_env_03/summaries",
            # list of labels, or 'all'
            labels=["graph", "entropy", "kl-divergence", "losses", "rewards"],
            frequency=100,  # store values every 100 timesteps
        ),
        saver=dict(
            directory='saved_models/agent_03_env_03/checkpoints',
            frequency=600  # save checkpoint every 600 seconds (10 minutes)
        ),
    )

    runner = Runner(agent=agent, environment=bad_seeds_environment)
    for _ in range(10):
        runner.run(num_episodes=10000)
        runner.run(num_episodes=1000, evaluation=True)

    bad_seeds_environment.close()
    agent.close()


if __name__ == "__main__":
    main()
