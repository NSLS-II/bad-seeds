from pathlib import Path

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from bad_seeds.simple.bad_seeds_04_bollux import Bollux


def main():

    bad_seeds_environment = Environment.create(
        environment=Bollux, seed_count=10, bad_seed_count=3, max_episode_length=100
    )

    agent = Agent.create(
        agent="ppo",
        # 113031
        # batch_size=10,
        # variable_noise=0.1,

        # 120700
        #batch_size=100,
        #variable_noise=0.1,

        #
        batch_size=1000,
        variable_noise=0.1,

        environment=bad_seeds_environment,
        summarizer=dict(
            directory="training_data/agent_ppo_01_env_bollux_1000/summaries",
            # list of labels, or 'all'
            labels=["graph", "entropy", "kl-divergence", "losses", "rewards"],
            frequency=100,  # store values every 100 timesteps
        ),
        # saver=dict(
        #     directory='saved_models/agent_04_env_04_1000/checkpoints',
        #     frequency=600  # save checkpoint every 600 seconds (10 minutes)
        # ),
    )

    runner = Runner(agent=agent, environment=bad_seeds_environment)
    for i in range(1000):
        print("running 1000 episodes")
        runner.run(num_episodes=1000)
        print("saving the agent")
        directory = Path(f"saved_models/agent_ppo_01_env_bollux/1000_{i}/checkpoints")
        if directory.exists():
            directory.rmdir()
        directory.mkdir(parents=True, exist_ok=True)
        agent.save(directory=str(directory), format="numpy")

    bad_seeds_environment.close()
    agent.close()


if __name__ == "__main__":
    main()
