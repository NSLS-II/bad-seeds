from pathlib import Path
import sys

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from bad_seeds.simple.bad_seeds_04_bollux import Bollux


def main():

    bad_seeds_environment = Environment.create(
        environment=Bollux,
        seed_count=10,
        bad_seed_count=3,
        max_episode_length=100,
        max_episode_timesteps=100,
    )

    batch_size = int(sys.argv[1])
    print(f"batch size: {batch_size}")

    #
    # l2_regularization = 0.1  123936
    # l2_regularization = 0.01 125106
    # l2_regularization = 0.001 125757
    # l2_regularization = 0.0001 133423 this one is learning fast,   average evaluation reward: 6.79
    l2_regularization = 0.0001

    agent = Agent.create(
        agent="ppo",

        # 112717 - batch size 10, saved best agent found in 5k episodes, average evaluation reward: 7.15
        # 112722 - batch size 20, saved best agent found in 5k episodes,
        batch_size=batch_size,
        variable_noise=0.1,

        l2_regularization=l2_regularization,

        # 140804  - 1000 episodes only
        # 151604  - catastrophic forgetting
        # 174356  - catastrophic forgetting
        # batch_size=10,
        # variable_noise=0.1,

        # 230046 -- 50000 episodes, save model at the end, catastrophic forgetting at 35k episodes
        # batch_size=50,
        # variable_noise=0.1,

        # 152501 - catastrophic forgetting at 55k episodes
        # batch_size=100,
        # variable_noise=0.1,

        # never tried
        # batch_size=200,
        # variable_noise=0.1,

        # never tried
        # batch_size=1000,
        # variable_noise=0.1,

        environment=bad_seeds_environment,
        summarizer=dict(
            directory=f"training_data/agent_ppo_01_env_bollux_bs{batch_size}_l2r{l2_regularization}/summaries",
            labels=["graph", "entropy", "kl-divergence", "losses", "rewards"],
            # frequency=100  not necessary?
        ),
        # this caused an exception
        # saver=dict(
        #    directory='saved_models/agent_ppo_01_env_bollux_1000/checkpoints',
        #    frequency=500
        # ),
    )

    saved_models_directory = Path("saved_models")
    saved_agent_directory = Path(f"agent_ppo_01_env_bollux_bs{batch_size}_l2r{l2_regularization}/checkpoints")

    export_directory = saved_models_directory / saved_agent_directory
    export_directory.mkdir(parents=True, exist_ok=True)

    best_average_evaluation_reward = 0.0
    runner = Runner(agent=agent, environment=bad_seeds_environment)
    training_episodes = 1000
    for i in range(1000):
        print(f"training for {training_episodes} episodes")
        runner.run(num_episodes=training_episodes)
        print("evaluating for 100 episodes")
        average_evaluation_reward = evaluate_agent(agent=agent, bad_seeds_env=bad_seeds_environment)
        print(f"  average evaluation reward: {average_evaluation_reward}")
        if average_evaluation_reward > best_average_evaluation_reward:
            print(f"saving model")
            best_average_evaluation_reward = average_evaluation_reward
            agent.save(directory=str(export_directory), filename="ppo_agent", format="numpy", append="episodes")

    bad_seeds_environment.close()
    agent.close()


def evaluate_agent(agent, bad_seeds_env):
    # take a break for evaluation
    sum_rewards = 0.0
    evaluation_episode_count = 100
    for _ in range(evaluation_episode_count):
        states = bad_seeds_env.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(
                states=states,
                internals=internals,
                independent=True,
                deterministic=True,
            )
            states, terminal, reward = bad_seeds_env.execute(
                actions=actions
            )
            sum_rewards += reward
    bad_seeds_env.reset()
    average_evaluation_reward = sum_rewards / evaluation_episode_count
    return average_evaluation_reward


if __name__ == "__main__":
    main()
