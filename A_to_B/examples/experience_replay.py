# Experience Replay

# Importing the libraries
import numpy as np
from collections import namedtuple, deque
import pickle
import time
import random

# Defining one Step
Step = namedtuple('Step', ['state', 'action', 'reward', 'next_state', 'done', 'metrics'])


# Making the AI progress on several (n_step) steps

class NStepProgress:

    def __init__(self, env, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.percentages = []
        self.env = env
        self.n_step = n_step

    def __iter__(self):
        state_rgb, route_distance, kmh = self.env.reset()
        metrics = np.array([route_distance, kmh], dtype=np.float32)
        history = deque()
        reward = 0.0
        eps = 0.20
        while True:
            action = self.ai(np.array([state_rgb]), metrics)[0][0]
            time.sleep(0.1)
            next_state, r, is_done, _, percentage, route_distance, kmh = self.env.step(action)
            reward += r
            metrics = np.array([route_distance, kmh], dtype=np.float32)
            history.append(Step(state=state_rgb, action=action, reward=r, next_state=next_state, done=is_done, metrics=metrics))
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)
            state_rgb = next_state
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.rewards.append(reward)
                self.percentages.append(percentage)
                reward = 0.0
                self.env.destroy_agents()
                self.env.reset()
                history.clear()

    def params_steps(self):
        rewards_steps = self.rewards
        percentage = self.percentages
        self.rewards = []
        self.percentages = []
        return rewards_steps, percentage


# Implementing Experience Replay

class ReplayMemory:

    def __init__(self, n_steps, capacity=10000, read_buffer=None):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        if read_buffer is False:
            self.buffer = deque()
        else:
            self.buffer = self.read_buffer()
            print("SUCCESS READ!")

    def sample_batch(self, batch_size):  # creates an iterator that returns random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs + 1) * batch_size <= len(self.buffer):
            yield vals[ofs * batch_size:(ofs + 1) * batch_size]
            ofs += 1

    def run_steps(self, samples):
        while samples > 0:
            entry = next(self.n_steps_iter)  # 10 consecutive steps
            self.buffer.append(entry)  # we put 200 for the current episode
            samples -= 1
        while len(self.buffer) > self.capacity:  # we accumulate no more than the capacity (10000)
            self.buffer.popleft()

    def save_buffer(self):
        pickle_out = open("buffer_semantic_scenario_2.pickle", "wb")
        pickle.dump(self.buffer, pickle_out)
        pickle_out.close()

    @staticmethod
    def read_buffer():
        print("READING PREVIEW BUFFER")
        pickle_in = open("buffer_semantic_scenario_2.pickle", "rb")
        buffer = pickle.load(pickle_in)
        time.sleep(2)
        return buffer