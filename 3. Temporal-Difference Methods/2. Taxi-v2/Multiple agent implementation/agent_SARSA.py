import numpy as np
from collections import defaultdict
import random

random.seed(a=47, version=2)


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.1
        self.gamma = 0.9
        self.num_episodes = 1
        self.epsilon = 0.15

    def get_epsilon_greedy_policy(self, state):
        policy = np.ones(self.nA) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        policy[best_action] = 1 - self.epsilon + (self.epsilon / self.nA)
        return policy

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.nA))

    def updated_q_sarsa(self, alpha, gamma, Q, nA, state, action, reward, next_state=None, next_action=None):
        Q_current = Q[state][action]
        Q_next = Q[next_state][next_action] if next_state is not None else 0
        target = reward + (gamma * Q_next)
        Q_updated = Q_current + (alpha * (target - Q_current))
        return Q_updated

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        next_action = np.argmax(self.Q[next_state])
        self.Q[state][action] += self.updated_q_sarsa(self.alpha, self.gamma, self.Q, self.nA, state, action, reward,
                                                      next_state, next_action)
        if done:
            self.num_episodes += 1
            self.epsilon = self.epsilon / self.num_episodes
