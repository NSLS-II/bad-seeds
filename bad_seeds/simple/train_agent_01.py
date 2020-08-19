from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from bad_seeds.simple import BadSeeds


def main():

    bad_seeds_environment = Environment.create(
        environment=BadSeeds, max_episode_timesteps=100,
    )

    agent = Agent.create(
        # agent="tensorforce",
        # update=64,
        # objective="policy_gradient",
        # reward_estimation=dict(horizon=max_turns),
        agent="a2c",
        batch_size=100,  # this seems to help a2c
        horizon=20,  # does this help a2c?
        exploration=0.01,  # tried without this at first
        l2_regularization=0.1,
        entropy_regularization=0.2,
        variable_noise=0.05,
        environment=bad_seeds_environment,
        summarizer=dict(
            directory="data/summaries",
            # list of labels, or 'all'
            labels=["graph", "entropy", "kl-divergence", "losses", "rewards"],
            frequency=100,  # store values every 100 timesteps
        ),
    )

    runner = Runner(agent=agent, environment=bad_seeds_environment)
    runner.run(num_episodes=100000)
    agent.save(directory="saved_models/agent_01_env_01")


if __name__ == "__main__":
    main()
