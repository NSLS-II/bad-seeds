import numpy as np
import tensorflow as tf
import random
from collections import deque
from tensorflow.keras.layers import Dense, Flatten


class DenseModel(tf.keras.Model):
    def __init__(self, input_shape, hidden_units, output_shape):
        super(DenseModel, self).__init__()
        self.input_layer = Flatten(input_shape=input_shape)
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(Dense(i, activation="relu", kernel_initializer="RandomNormal"))
        self.output_layer = Dense(output_shape, activation="linear", kernel_initializer="RandomNormal")

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


def sequential_model(input_shape, hidden_units, output_shape, lr):
    model = tf.keras.models.Sequential()
    model.add(Flatten(input_shape=input_shape))
    for i in hidden_units:
        model.add(Dense(i, activation="relu", kernel_initializer="RandomNormal"))
    model.add(Dense(output_shape, activation="linear"))
    model.compile(loss="mse", optimizer=tf.optimizers.Adam(lr=lr))
    return model


class DQN:
    def __init__(self, state_shape, num_actions, *, gamma=0.95, exploration=0.0, exploration_decay=0.995,
                 exploration_min=0.1, experience_min=0, experience_max=100000, batch_size=32, dense_layers=(24, 24),
                 learning_rate=0.001, copy_step=1):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.exploration_rate = exploration
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.memory = deque(maxlen=experience_max)
        self.experience_min = max(experience_min, batch_size)
        self.batch_size = batch_size
        self.copy_step = copy_step
        # self.training_model = DenseModel(self.state_shape, dense_layers,
        #                                  self.num_actions)  # Updated during update schedule
        # self.target_model = DenseModel(self.state_shape, dense_layers,
        #                                self.num_actions)  # Held for predictions during an episode
        # self.optimizer = tf.optimizers.Adam(learning_rate)
        self.training_model = sequential_model(self.state_shape, dense_layers, self.num_actions, learning_rate)
        self.target_model = sequential_model(self.state_shape, dense_layers, self.num_actions, learning_rate)

    def train_predict(self, inputs):
        return self.training_model.predict(inputs)

    def target_predict(self, inputs):
        return self.target_model.predict(inputs)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.num_actions)
        q_values = self.train_predict(state)
        return np.argmax(q_values[0])

    def train(self):
        """Experience replay"""
        if len(self.memory) < self.batch_size:
            return 0
        batch = list(zip(*random.sample(self.memory, self.batch_size)))
        state, action, reward, state_next, terminal = batch
        state = np.stack(state)
        state_next = np.stack(state_next)
        reward = np.array(reward)
        terminal = np.array(terminal)
        value_next = np.amax(self.target_predict(state_next), axis=1)
        q_update = np.where(terminal, reward, reward + self.gamma * value_next)
        q_values = self.train_predict(state)
        q_values[tuple(range(len(action))), action] = q_update
        history = self.training_model.fit(state, q_values, epochs=1, verbose=0)
        # with tf.GradientTape() as tape:
        #     q_values = tf.math.reduce_sum(
        #         self.train_predict(state) * tf.one_hot(action, self.num_actions), axis=1)
        #     loss = tf.math.reduce_mean(tf.square(q_update - q_values))
        # variables = self.training_model.trainable_variables
        # gradients = tape.gradient(loss, variables)
        # self.optimizer.apply_gradients(zip(gradients, variables))
        return history.history['loss']

    def copy_weights(self):
        vars1 = self.target_model.trainable_variables
        vars2 = self.training_model.trainable_variables
        for v1, v2 in zip(vars1, vars2):
            v1.assign(v2)

    def episode(self, env):
        rewards = list()
        step_count = 0
        terminal = False
        state = env.reset()
        losses = list()
        while not terminal:
            action = self.act(state[np.newaxis, :])
            prev_state = state
            state, terminal, reward = env.execute(actions=action)
            if terminal:
                env.reset()
            rewards.append(reward)
            self.remember(prev_state, action, reward, state, terminal)
            loss = self.train()
            if isinstance(loss, int):
                losses.append(loss)
            elif isinstance(loss, list):
                losses.append(loss[-1])
            else:
                losses.append(loss.numpy())
            step_count += 1
            if step_count % self.copy_step == 0:
                self.copy_weights()
        return np.sum(rewards), np.mean(losses)

    def play(self, env, n_episodes, tensorboard=True, verbose=False):
        import datetime
        if tensorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = 'training_data/dqn/' + current_time
            summary_writer = tf.summary.create_file_writer(log_dir)

        total_rewards = np.empty(n_episodes)
        for n in range(n_episodes):
            self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)
            total_reward, losses = self.episode(env)
            total_rewards[n] = total_reward
            avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
            if tensorboard:
                with summary_writer.as_default():
                    tf.summary.scalar('episode reward', total_reward, step=n)
                    tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
                    tf.summary.scalar('average loss)', losses, step=n)
                    summary_writer.flush()

            if verbose and (n % 10 == 0):
                print("episode:", n,
                      "episode reward:", total_reward,
                      "explore:", self.exploration_rate,
                      "avg reward (last 100):", avg_rewards,
                      "episode loss: ", losses)
        env.close()


def main():
    from tensorforce.environments import Environment
    from bad_seeds.simple.bad_seeds_02 import BadSeeds02
    from bad_seeds.simple.tf_utils import tensorflow_settings
    tensorflow_settings()
    env = Environment.create(
        environment=BadSeeds02, seed_count=10, bad_seed_count=3, history_block=2, max_episode_timesteps=300
    )
    agent = DQN(env.states()['shape'], env.environment.seed_count, exploration=0.25)
    agent.play(env, 5000, verbose=True)


if __name__ == "__main__":
    main()
