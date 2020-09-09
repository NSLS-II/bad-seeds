from pathlib import Path

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from bad_seeds.simple.bad_seeds_04_bollux import Bollux


def main():

    bad_seeds_environment = Environment.create(
        environment=Bollux, seed_count=10, bad_seed_count=3, max_episode_length=100
    )

    # 20200820-223031
    # 20200820-233243

    # batch_size 1000 goes not get smarter or dumber
    # batch_size 100 20200821-095410 gets dumber
    # try batch size 10000 !

    agent = Agent.create(
        agent="a2c",
        batch_size=10000,  # changed for 04 but was this a mistake? no
        horizon=50,     # changed from 100 to 50 for agent_04
        discount=0.97,  # new for agent_04
        #exploration=0.05,  # turned off for agent_04 - turn on for 05?
        l2_regularization=0.1,
        #entropy_regularization=0.2,  # turned off for agent_03
        variable_noise=0.5,  # changed from 0.1 to 0.5 for agent_04
        environment=bad_seeds_environment,
        summarizer=dict(
            directory="training_data/agent_04_bollux_1000000/summaries",
            # list of labels, or 'all'
            labels=["graph", "entropy", "kl-divergence", "losses", "rewards"],
            frequency=100,  # store values every 100 timesteps
        ),
        saver=dict(
            directory='saved_models/agent_04_bollux_1000000/checkpoints',
            frequency=6000  # save checkpoint every 6000 seconds (100 minutes)
        ),
    )

    # this is the batch_size = 10000 version
    # I hope it is the last env 04
    runner = Runner(agent=agent, environment=bad_seeds_environment)
    runner.run(num_episodes=1000000)
    #for i in range(100):
    #    print("running 10000 episodes")
    #    runner.run(num_episodes=10000)
    #    print("saving the agent")
    #    directory = Path(f"saved_models/agent_04_env_04_1000000/10000_{i}/checkpoints")
    #    if directory.exists():
    #        directory.rmdir()
    #    directory.mkdir(parents=True, exist_ok=True)
    #    agent.save(directory=str(directory), format="numpy")

    bad_seeds_environment.close()
    agent.close()


if __name__ == "__main__":
    main()
