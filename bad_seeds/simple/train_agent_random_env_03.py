from pathlib import Path

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from bad_seeds.simple.bad_seeds_03 import BadSeeds03


def main():

    bad_seeds_environment = Environment.create(
        environment=BadSeeds03, seed_count=10, bad_seed_count=3, max_episode_length=100
    )

    agent = Agent.create(
        agent="random",
        environment=bad_seeds_environment,
        summarizer=dict(
            directory="training_data/agent_random_env_03/summaries",
            # list of labels, or 'all'
            labels=["graph", "rewards"],
            frequency=100,  # store values every 100 timesteps
        ),
    )

    runner = Runner(agent=agent, environment=bad_seeds_environment)
    runner.run(num_episodes=10000)

    bad_seeds_environment.close()
    agent.close()

    #215242


if __name__ == "__main__":
    main()
