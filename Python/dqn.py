import gym

env = gym.make("CartPole-v1")

import matplotlib.pyplot as plt
import math
import numpy as np
import random

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam

np.random.seed(0)
random.seed(0)

RENDER_STEPS = 25
SESSIONS = 100

EXPLORATION_RATE = 1.0
EXPLORATION_RATE_DECAY = 0.995
EXPLORTATION_RATE_MIN = 0.01

BUFFER_SIZE = 10


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
        self._forgetfullness = 0.00

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


class DQNetwork:
    def __init__(
        self, action_space, observation_space, exploration_rate, exploration_rate_decay
    ):
        self._action_space = action_space
        self._observation_space = observation_space
        self._exploration_rate = exploration_rate
        self._exploration_rate_decay = exploration_rate_decay
        self._discount = 0.95

        self.model = Sequential()
        self.model.add(
            Dense(24, input_shape=observation_space.shape, activation="relu")
        )
        self.model.add(Dense(24, activation="linear"))
        self.model.add(Dense(self._action_space.n, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=1e-3))

    def act(self, observation):
        if np.random.rand() <= self._exploration_rate:
            return self._action_space.sample()

        return np.argmax(self.model.predict(observation))

    def learn_from_experience(self, experience):
        self._exploration_rate *= self._exploration_rate_decay

        for observation, next_observation, action, reward, done in experience:
            update = reward + self._discount * np.amax(
                self.model.predict(next_observation)[0]
            )
            if done:
                update = reward

            target = self.model.predict(observation)
            target[0][action] = update

            self.model.fit(observation, target, verbose=0)


plotter = Plotter()
replay_experience = ReplayExperience(BUFFER_SIZE)
network = DQNetwork(
    env.action_space, env.observation_space, EXPLORATION_RATE, EXPLORATION_RATE_DECAY
)

for session in range(1, SESSIONS + 1):  # offset by 1 for better formatting in results
    done = False
    step = 0
    experience = []

    observation = env.reset()
    observation = observation.reshape(1, -1)

    while not done:
        if session % RENDER_STEPS == 1:
            env.render()
        step += 1

        action = network.act(observation)
        next_observation, reward, done, _ = env.step(action)
        next_observation = next_observation.reshape(1, -1)

        experience.append((observation, next_observation, action, reward, done))
        observation = next_observation

    network.learn_from_experience(experience)  # Learns from immediate experience
    replay_experience.make_experience(step, experience)  # Learns from past experience

    for _ in range(5):
        experience = replay_experience.get_experience()
        network.learn_from_experience(experience)

    print(f"Session {session} took {step} steps.")
    plotter.add_plot_pair(session, step)

env.close()
plotter.plot_smooth_graph(SESSIONS // 10)
