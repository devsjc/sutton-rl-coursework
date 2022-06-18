"""
Assesses the ineffectiveness of a sample-average stepsize function
on non-stationary problems, i.e. problems where the true reward
value changes over time for each action
"""

from typing import Callable, List, NamedTuple, Tuple
import random
import numpy as np
import matplotlib.pyplot as plt

K = 10
EPSILON = 0.1
STEPS = 20000
PROBLEMS = 200


class NonstationaryBandit:
    def __init__(self):
        self.true_action_values = [0 for _ in range(K)]

    def get_reward(self, action_number: int, action_counts: List[int]) -> Tuple[float, List[int]]:
        # get a reward value for the input action
        reward = np.random.normal(loc=self.true_action_values[action_number])
        # take a random walk step on all the action values
        self.true_action_values = [v + random.gauss(mu=0, sigma=0.01) for v in self.true_action_values]
        action_counts[action_number] += 1
        return reward, action_counts


class ActionValue(NamedTuple):
    count: int
    estimate: float


# stepsize_constant defines the stepsize function for
# constant stepsize
def stepsize_constant(count: int) -> float:
    return 0.1 if count > 0 else 0


# stepsize_sampleavg defines the stepsize function for
# sample-averaged stepsize
def stepsize_sampleavg(count: int) -> float:
    return 1/count if count > 0 else 0


# action_value_method updates the estimated action values
# according to the newly received reward, the stepsize function
def action_value_method(action_estimates: List[float],
                        action_counts: List[int],
                        reward: float, action_number: float,
                        stepsize_function: Callable[[int], float]) -> List[float]:

    new_estimates = [est_count[0] + (stepsize_function(est_count[1]) * (reward - est_count[0])) if i == action_number
                     else est_count[0]
                     for i, est_count in enumerate(zip(action_estimates, action_counts))]

    return new_estimates


# select_action defines our action selection rule. In this case ε-greedy.
def select_action(action_estimates: List[float]) -> int:
    # take exploratory step with probability epsilon
    if np.random.normal() <= EPSILON:
        return np.random.randint(K)
    # otherwise, take the action with the highest estimated value
    else:
        max_value_actions = np.argwhere(action_estimates == np.amax(action_estimates)).flatten().tolist()
        # if there is more than one highest value, select one at random
        greedy_action = random.choice(max_value_actions)
        return greedy_action


# take_step carries out one step of a run on a given problem
def take_step(bandit: NonstationaryBandit,
              stepsize_function: Callable[[int], float],
              action_counts: List[int],
              action_estimates: List[float]) -> Tuple[List[float], List[int], float]:

    # choose an action according to the current action estimates
    # using the ε-greedy approach
    action_number = select_action(action_estimates)
    # get a reward for that action (in turn updating the action counts
    # and carrying a random walk step on the action values
    reward, action_counts = bandit.get_reward(action_number, action_counts)
    # update the action estimates using our action_value_method
    # for the new reward value received
    action_estimates = action_value_method(action_estimates, action_counts, reward, action_number, stepsize_function)

    return action_estimates, action_counts, reward


def main():

    for func in [stepsize_constant, stepsize_sampleavg]:
        sum_rewards = [0 for _ in range(STEPS)]
        # loop over all the bandit problems
        for problem in range(PROBLEMS):
            bandit = NonstationaryBandit()
            action_estimates: List[float] = [0 for _ in range(K)]
            action_counts: List[int] = [0 for _ in range(K)]
            # perform a run of STEPS steps on the current bandit problem
            # update the sum_rewards list by adding to it the reward
            # attained at each step
            for step in range(STEPS):
                action_estimates, action_counts, received_reward = take_step(
                    bandit=bandit,
                    stepsize_function=func,
                    action_counts=action_counts,
                    action_estimates=action_estimates)

                sum_rewards[step] = sum_rewards[step] + received_reward

        # plot each epsilon value as a line graph
        avg_rewards = [sum_rewards[i] / PROBLEMS for i in range(STEPS)]
        plt.plot([i for i in range(STEPS)], avg_rewards, label=func.__str__())

    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

