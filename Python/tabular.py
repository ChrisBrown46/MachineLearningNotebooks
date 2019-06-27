import gym

env = gym.make("CartPole-v1")

import matplotlib.pyplot as plt
import math
import numpy as np
import random

RENDER_STEPS = 250
SESSIONS = 500

BINS = 20
REWARD_DISCOUNT = 0.98
LEARNING_RATE_DECAY = 0.995
LEARNING_RATE_MIN = 0.150
EXPLORATION_RATE = 1.0
EXPLORATION_RATE_DECAY = 0.975
EXPLORTATION_RATE_MIN = 0.005

BUFFER_SIZE = 20


class Plotter:
    def __init__(self):
        self._x_values = []
        self._y_values = []

    def add_plot_pair(self, x, y):
        self._x_values.append(x)
        self._y_values.append(y)

    def plot(self):
        plt.plot(self._x_values, self._y_values)
        plt.show()

    def plot_and_save(self, file_name):
        plt.plot(self._x_values, self._y_values)
        plt.savefig(file_name)

    def plot_smooth_graph(self, averaging_filter_length=100):
        if averaging_filter_length > len(self._x_values):
            self.plot()
            return

        rolling_average = []
        for index in range(len(self._x_values)):
            min_index = index - (averaging_filter_length / 2)
            min_index = int(0 if min_index < 0 else min_index)
            max_index = index + (averaging_filter_length / 2)
            max_index = int(
                len(self._x_values) - 1
                if max_index >= len(self._x_values)
                else max_index
            )

            rolling_average.append(
                sum(self._y_values[min_index:max_index]) // (max_index - min_index)
            )

        plt.title("Steps Achieved Per Session Over Training Duration")
        plt.xlabel("Sessions")
        plt.ylabel("Steps Achieved Per Session")
        plt.plot(
            self._x_values,
            rolling_average,
            color="#ade6bb",
            linewidth=2.5,
            label=f"Running Average of {averaging_filter_length} Sessions",
        )
        plt.plot(
            self._x_values,
            self._y_values,
            color="#e6add8",
            linewidth=0.25,
            label=f"Raw Steps Achieved",
        )
        plt.show()


class ReplayExperience:
    def __init__(self, buffer_size):
        self._buffer = []
        self._buffer_size = buffer_size
        self._forgetfullness = 0.15

    def get_experience(self):
        return random.choice(self._buffer)[1]

    def make_experience(self, value, experience):
        if len(self._buffer) < self._buffer_size:
            self._buffer.append((value, experience))
            return

        if np.random.rand() <= self._forgetfullness:
            del self._buffer[np.random.randint(len(self._buffer))]

        # Find the worst memory and replace it
        min_value = value
        min_index = -1
        for index in range(len(self._buffer)):
            if self._buffer[index][0] < min_value:
                min_value = self._buffer[index][0]
                min_index = index
        if min_index != -1:
            self._buffer[min_index] = (value, experience)


class Table:
    def __init__(
        self,
        action_space,
        observation_space,
        bins,
        reward_dicount,
        exploration_rate,
        exploration_rate_min,
        exploration_rate_decay,
        learning_rate_min,
        learning_rate_decay,
    ):
        self._action_space = action_space
        self._observation_space = observation_space
        self._bins = bins
        self._reward_discount = reward_dicount
        self._exploration_rate = exploration_rate
        self._exploration_rate_min = exploration_rate_min
        self._exploration_rate_decay = exploration_rate_decay
        self._learning_rates = np.ones(
            (bins ** observation_space.shape[0], action_space.n)
        )
        self._learning_rate_min = learning_rate_min
        self._learning_rate_decay = learning_rate_decay

        self._table = np.zeros((bins ** observation_space.shape[0], action_space.n))

    def act(self, observation):
        observation = self._scale_observation(observation)
        bin = self._bin_observation(observation)

        if np.random.rand() <= self._exploration_rate:
            return self._action_space.sample()

        return np.argmax(self._table[bin])

    def learn_from_experience(self, experience):
        penalty = -sum(step[2] for step in experience)
        experience_length = len(experience)
        discount = self._reward_discount

        for index in range(experience_length - 1, 0, -1):
            observation, action, _ = experience[index]
            observation = self._scale_observation(observation)
            bin = self._bin_observation(observation)

            self._table[bin][action] += (
                self._learning_rates[bin][action] * discount * penalty
            )

            discount *= self._reward_discount
            self._update_learning_rate(bin, action)

        self._update_exploration_rate()

    def _update_learning_rate(self, bin, action):
        self._learning_rates[bin][action] *= self._learning_rate_decay
        self._learning_rates[bin][action] = max(
            self._learning_rate_min, self._learning_rates[bin][action]
        )

    def _update_exploration_rate(self):
        self._exploration_rate *= self._exploration_rate_decay
        self._exploration_rate = max(self._exploration_rate_min, self._exploration_rate)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _scale_observation(self, observation):
        scaled_observation = []

        for index in range(len(self._observation_space.low)):
            high = self._sigmoid(self._observation_space.high[index])
            low = self._sigmoid(self._observation_space.low[index])

            scaled_value = self._sigmoid(observation[index])
            scaled_value = (scaled_value - low) / (high - low)
            normalized_value = scaled_value * self._bins

            scaled_observation.append(int(normalized_value))

        return scaled_observation

    def _bin_observation(self, observation):
        bin = 0
        for index in range(len(observation)):
            bin += (2 ** index) * observation[index]
        return int(bin)  # must be an int since it is used as an index


plotter = Plotter()
replay_experience = ReplayExperience(BUFFER_SIZE)
table = Table(
    env.action_space,
    env.observation_space,
    BINS,
    REWARD_DISCOUNT,
    EXPLORATION_RATE,
    EXPLORTATION_RATE_MIN,
    EXPLORATION_RATE_DECAY,
    LEARNING_RATE_MIN,
    LEARNING_RATE_DECAY,
)

for session in range(1, SESSIONS + 1):  # offset by 1 for better formatting in results
    done = False
    step = 0
    experience = []

    if session == 1:
        env = gym.wrappers.Monitor(env, "./Video")
        x = input("Enter to view last thingy")
        table._exploration_rate = 0
        table._exploration_rate_min = 0
        table._learning_rate_min = 0
        table._learning_rate_decay = 0

    observation = env.reset()

    while not done:

        if session % RENDER_STEPS == 1:
            env.render()
        step += 1

        action = table.act(observation)
        observation, reward, done, _ = env.step(action)

        experience.append((observation, action, reward))

    if step == 500:
        table._exploration_rate = 0
        table._exploration_rate_min = 0

    table.learn_from_experience(experience)
    replay_experience.make_experience(step, experience)
    experience = replay_experience.get_experience()
    table.learn_from_experience(experience)

    print(f"Session {session} took {step} steps.")
    plotter.add_plot_pair(session, step)

env.close()
plotter.plot_smooth_graph()
