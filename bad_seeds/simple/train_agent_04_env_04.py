from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from bad_seeds.simple.bad_seeds_04 import BadSeeds04


def main():

    bad_seeds_environment = Environment.create(
        environment=BadSeeds04, seed_count=10, bad_seed_count=3, max_episode_length=100
    )

    # 20200820-223031
    # 20200820-233243

    agent = Agent.create(
        agent="a2c",
        batch_size=100,  # changed for 04 but was this a mistake?
        horizon=50,     # changed from 100 to 50 for agent_04
        discount=0.97,  # new for agent_04
        #exploration=0.05,  # turned off for agent_04 - turn on for 05?
        l2_regularization=0.1,
        #entropy_regularization=0.2,  # turned off for agent_03
        variable_noise=0.5,  # changed from 0.1 to 0.5 for agent_04
        environment=bad_seeds_environment,
        summarizer=dict(
            directory="training_data/agent_04_env_04_1/summaries",
            # list of labels, or 'all'
            labels=["graph", "entropy", "kl-divergence", "losses", "rewards"],
            frequency=100,  # store values every 100 timesteps
        ),
        saver=dict(
            directory='saved_models/agent_04_env_04_1/checkpoints',
            frequency=600  # save checkpoint every 600 seconds (10 minutes)
        ),
    )

    runner = Runner(agent=agent, environment=bad_seeds_environment)

    runner.run(num_episodes=10000)

    bad_seeds_environment.close()
    agent.close()


if __name__ == "__main__":
    main()
