"""
Looking at using Bayes rule to estimate bandit values.

Idea taken from https://lazyprogrammer.me/bayesian-bandit-tutorial/

for number_of_trials:
  take random sample from each bandit with its current (a, b)
  for the bandit with the largest sample, pull its arm
  update that bandit with the data from the last pull

"""
import random

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from typing import NamedTuple, List

NUM_TRIALS = 200
REAL_PS = [0.2, 0.5, 0.75]


class BetaParams(NamedTuple):
    a: float
    b: float


def main():
    # initialise our estimates as normal distributions
    bandit_estimates: List[BetaParams] = [BetaParams(1, 1) for _ in REAL_PS]
    for i in range(NUM_TRIALS):
        # Take sample samples from each bandit with our current estimates of the beta params
        samples = [np.random.beta(a, b)for a, b in bandit_estimates]
        # choose the bandit with the largest sample size to take an action from
        chosen_bandit = np.argmax(samples)
        reward = np.random.binomial(1, REAL_PS[chosen_bandit])
        # Update that bandit's parameter estimates according to the reward
        bandit_estimates[chosen_bandit] = BetaParams(
            bandit_estimates[chosen_bandit][0] + reward,
            bandit_estimates[chosen_bandit][1] + 1 - reward)

    x = np.linspace(0, 1, 200)
    for i, pms in enumerate(bandit_estimates):
        y = stats.beta.pdf(x, a=pms.a, b=pms.b)
        plt.plot(x, y, label=f"p={REAL_PS[i]}")

    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
