"""
Assessing the effectiveness of greedy and ε-greedy action value methods.

Creates a testbed consisting of a set of 2000 k-armed bandits with k=10. Each bandit problem has action values q*
selected from a gaussian distribution of mean 0 and variance 1. Then an agent is run for 1000 steps on each problem,
and its learning behaviour analyzed.
"""

import random

from typing import List, Tuple, NamedTuple
import numpy as np
import matplotlib.pyplot as plt

# --- Define constants for the testbed
K = 10  # number of arms on bandit
STEPS = 1000  # number of learning steps taken per problem run
PROBLEMS = 2000  # number of bandit problems to be solved
DEFAULT_ACTION_VALUE_ESTIMATE = 0  # default estimated action value


# ActionValue holds all the data required to estimate the value of
# an action using the sample average method
class ActionValue(NamedTuple):
    sum: float
    num: int

    # estimate_value gets an estimation for the expected reward received
    # from carrying out the action, using the sample-average method:
    # the mean of all the rewards previously received for that action
    def estimate_value(self) -> float:
        return self.sum / self.num if self.num > 0 else DEFAULT_ACTION_VALUE_ESTIMATE


# Bandit creates an instance of a bandit with K arms, each with true
# action values selected from a normal distribution
class Bandit:
    def __init__(self):
        self.true_action_values: List = np.random.normal(size=(K,)).tolist()

    # reward Rt for an action is selected from a normal distribution
    # with mean q* of that action and variance 1
    def get_reward(self, action_number: int) -> float:
        if action_number >= K:
            raise Exception(f"Can't access action with number {action_number} for a {K} armed bandit")
        return np.random.normal(loc=self.true_action_values[action_number])


# select_action defines our action selection rule. In this case ε-greedy.
def select_action(epsilon: float, action_values: List[ActionValue]) -> int:
    # take exploratory step with probability epsilon
    if np.random.normal() <= epsilon:
        return np.random.randint(K)
    # otherwise, take the action with the highest estimated value
    else:
        action_value_estimates = [av.estimate_value() for av in action_values]
        max_value_actions = np.argwhere(action_value_estimates == np.amax(action_value_estimates)).flatten().tolist()
        # if there is more than one highest value, select one at random
        greedy_action = random.choice(max_value_actions)
        return greedy_action


# update_action_value_estimates keeps track of the sum of our rewards and the number of
# times each reward has been chosen
def update_action_value_estimates(action_value_estimates: List[ActionValue], reward_value: float,
                                  action_number: int) -> List[ActionValue]:
    new_estimates = [av if i != action_number
                     else ActionValue(av.sum + reward_value, av.num + 1)
                     for i, av in enumerate(action_value_estimates)]
    return new_estimates


# take_step carries out a single step of the learning process
def take_step(bandit: Bandit, epsilon: float, action_value_estimates: List[ActionValue]) \
        -> Tuple[List[ActionValue], float]:
    # choose an action according to our action selection method
    action_number = select_action(epsilon, action_value_estimates)
    # get a reward according to our reward distribution
    attained_reward = bandit.get_reward(action_number)
    # keep track of the attained rewards and update action value estimates
    new_av_estimates = update_action_value_estimates(action_value_estimates, attained_reward, action_number)

    return new_av_estimates, attained_reward


def main():

    for epsilon in [0, 0.01, 0.1]:
        sum_rewards = [0 for _ in range(STEPS)]
        # loop over all the bandit problems
        for problem in range(PROBLEMS):
            action_value_estimates: List[ActionValue] = [ActionValue(0, 0) for _ in range(K)]
            bandit = Bandit()
            # perform a run of STEPS steps on the current bandit problem
            # update the sum_rewards list by adding to it the reward
            # attained at each step
            for step in range(STEPS):
                action_value_estimates, received_reward = take_step(
                    bandit=bandit,
                    epsilon=epsilon,
                    action_value_estimates=action_value_estimates)

                sum_rewards[step] = sum_rewards[step] + received_reward

        # plot each epsilon value as a line graph
        avg_rewards = [sum_rewards[i] / PROBLEMS for i in range(STEPS)]
        plt.plot([i for i in range(STEPS)], avg_rewards, label=epsilon.__str__())

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
