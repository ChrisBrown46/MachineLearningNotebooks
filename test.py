import gym

env = gym.make("CartPole-v1")

import matplotlib.pyplot as plt
import numpy as np
import random

RENDER_STEPS = 10_000
SESSIONS = 100_000

BINS = 5
REWARD_DISCOUNT = 0.995
MAX_LEARNING_RATE = 1.000
MIN_LEARNING_RATE = 0.150
INITIAL_EPSILON = 0.950
MIN_EPSILON = 0.005

BUFFER_SIZE = 15


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
        learning_rate,
        min_learning_rate,
        reward_discount,
        initial_epsilon,
        min_epsilon,
    ):
        self._action_space = action_space
        self._observation_space = observation_space
        self._bins = bins
        self._learning_rate = learning_rate
        self._min_learning_rate = min_learning_rate
        self._reward_discount = reward_discount
        self._epsilon = initial_epsilon
        self._min_epsilon = min_epsilon

        self._table = np.zeros((bins ** observation_space.shape[0], action_space.n))

        self._sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self._transform = lambda x: int(self._sigmoid(x) * bins)

    def act(self, observation):
        if np.random.rand() % 100 <= self._epsilon * 100:
            return self._action_space.sample()

        bin = self._bin_observation(observation)
        return np.argmax(self._table[bin])

    def observe(self, last_observation, observation, action, reward, step):
        last_bin = self._bin_observation(last_observation)
        bin = self._bin_observation(observation)

        self._table[last_bin][action] = self._learning_rate * (
            reward + (self._reward_discount ** step) * np.max(self._table[bin])
        )

    def update_learning_rate(self, decay):
        self._epsilon -= 2.0 * decay
        self._epsilon = max(self._min_epsilon, self._epsilon)
        self._learning_rate -= decay
        self._learning_rate = max(self._min_learning_rate, self._learning_rate)

    def _bin_observation(self, observation):
        bin = 0
        for index in range(len(observation)):
            bin += (2 ** index) * self._transform(observation[index])
        return int(bin)  # must be an int since it is used as an index


plotter = Plotter()
replay_experience = ReplayExperience(BUFFER_SIZE)
table = Table(
    env.action_space,
    env.observation_space,
    BINS,
    MAX_LEARNING_RATE,
    MIN_LEARNING_RATE,
    REWARD_DISCOUNT,
    INITIAL_EPSILON,
    MIN_EPSILON,
)
decay = 1 / SESSIONS

for session in range(1, SESSIONS + 1):  # offset by 1 for better formatting in results
    done = False
    step = 0
    experience = []

    observation = env.reset()
    table.update_learning_rate(decay)

    while not done:
        if session % RENDER_STEPS == 1:
            env.render()
        step += 1

        last_observation = observation
        action = table.act(observation)
        observation, reward, done, _ = env.step(action)
        table.observe(last_observation, observation, action, reward, step)
        experience.append((last_observation, observation, action, reward, step))

    replay_experience.make_experience(step, experience)
    memory = replay_experience.get_experience()
    for experience in memory:
        table.observe(
            experience[0], experience[1], experience[2], experience[3], experience[4]
        )

    print(
        f"Session {session} took {step} steps and has a learning rate of {table._learning_rate:.5f} and epsilon of {table._epsilon:.5f}."
    )
    plotter.add_plot_pair(session, step)


env.close()
plotter.plot()
