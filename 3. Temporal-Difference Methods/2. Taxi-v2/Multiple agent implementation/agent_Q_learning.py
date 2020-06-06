import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.04
        self.gamma = 0.8
        self.num_episodes = 1
        self.epsilon = 1
    
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
        policy = self.get_epsilon_greedy_policy(state)
        return np.random.choice(self.nA, p=policy)
   
    def updated_q_expected_sarsa(self, alpha, gamma, Q, nA, state, action, reward, next_state=None):
        Q_current = Q[state][action]
        epsilon_policy = self.get_epsilon_greedy_policy(next_state) if next_state is not None else 0
        Q_next = np.dot(Q[next_state], epsilon_policy)
        target = reward + ( gamma * Q_next)
        Q_updated = Q_current + ( alpha * ( target - Q_current ) )
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
        self.Q[state][action] += self.updated_q_expected_sarsa(self.alpha, self.gamma, self.Q, self.nA, state, action, reward, next_state)
        if done:
            self.num_episodes += 1
            self.epsilon = 1/self.num_episodes
            self.gamma = self.gamma 
