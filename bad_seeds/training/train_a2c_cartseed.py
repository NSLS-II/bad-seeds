from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from bad_seeds.environments.cartseed import CartSeed, CartSeedCountdown
from bad_seeds.utils.tf_utils import tensorflow_settings
from pathlib import Path


def set_up(timelimit=50,
           scoring=None,
           gpu_idx=0,
           batch_size=16,
           env_version=1,
           seed_count=10.,
           out_path=None):
    """
    Set up a rushed CartSeed agent with less time than it needs to complete an episode.
    Parameters
    ----------
    timelimit : int
        Turn time limit for episode
    scoring : str in {'t22', 'tt5', 'monotonic', 'linear', 'square', 'default'
        Name of reward function
    gpu_idx : int
        optional index for GPU
    batch_size : int
        Batch size for training
    env_version : int in {1, 2}
        Environment version. 1 being ideal time, 2 being time limited
    seed_count : int
        Number of bad seeds
    out_path : path
        Toplevel dir for output of models and checkpoints

    Returns
    -------
    Environment
    Agent
    """

    def tt2(state, *args):
        if state[1] >= 5:
            return 2
        else:
            return 1

    def tt5(state, *args):
        if state[1] >= 5:
            return 5
        else:
            return 1

    def monotonic(state, *args):
        # This worked but would be better described as heavyside linear
        return float(state[1] > 5) * state[1]

    def linear(state, *args):
        return state[1]

    def square(state, *args):
        return state[1] ** 2

    def default(state, *args):
        return 1

    func_dict = dict(tt2=tt2,
                     tt5=tt5,
                     monotonic=monotonic,
                     linear=linear,
                     square=square,
                     default=default)

    tensorflow_settings(gpu_idx)
    if out_path is None:
        out_path = Path().absolute()
    else:
        out_path = Path(out_path).expanduser().absolute()
    if env_version == 1:
        environment = CartSeed(seed_count=seed_count,
                               bad_seed_count=None,
                               max_count=10,
                               sequential=True,
                               revisiting=True,
                               bad_seed_reward_f=func_dict.get(scoring, None),
                               measurement_time=timelimit)
    elif env_version == 2:
        environment = CartSeedCountdown(seed_count=seed_count,
                                        bad_seed_count=None,
                                        max_count=10,
                                        sequential=True,
                                        revisiting=True,
                                        bad_seed_reward_f=func_dict.get(scoring, None),
                                        measurement_time=timelimit)
    else:
        raise NotImplementedError
    env = Environment.create(environment=environment)
    agent = Agent.create(
        agent="a2c",
        batch_size=batch_size,
        environment=env,
        summarizer=dict(
            directory=out_path / "training_data/a2c_cartseed/{}_{}_{}_{}".format(env_version, timelimit, scoring, batch_size),
            labels="all",
            frequency=1,
        ),
    )

    return env, agent


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


def main():
    env, agent = set_up(timelimit=None,
                        scoring='default',
                        batch_size=128,
                        gpu_idx=0,
                        env_version=2)
    runner = Runner(agent=agent, environment=env)
    runner.run(num_episodes=int(3 * 10 ** 3))
    agent.save(directory="saved_models")
    agent.close()
    env.close()


if __name__ == "__main__":
    main()
