"""
Assesses the ineffectiveness of a sample-average stepsize function
on non-stationary problems, i.e. problems where the true reward
value changes over time for each action
"""

from typing import Callable, List, NamedTuple
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.1
STEPS = 10000


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
#
def action_value_method(action_estimates: List[float],
                        reward: float, action_number: float,
                        stepsize_function: Callable[[int], float],
                        step: int) -> List[float]:

    new_estimates = [est + (stepsize_function(step) * (reward - est)) if i == action_number
                     else est
                     for i, est in enumerate(action_estimates)]

    return new_estimates
