from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, k: int):
        self.k = k
        self.true_action_values = np.random.normal(size=(k,))

    def get_reward(self, action_number: int) -> float:
        if action_number > self.k:
            raise Exception(f"Can't access action with number {action_number} for a {self.k} armed bandit")
        return np.random.normal(loc=self.true_action_values[action_number])

    def get_optimal_action(self) -> int:
        return np.max(self.true_action_values)


class Agent:
    def __init__(self, epsilon: float, bandit: Bandit):
        self.epsilon: float = epsilon
        self.bandit: Bandit = bandit
        # Initialise all action estimates as 0
        self.action_value_estimates: [] = np.zeros(shape=(self.bandit.k,))
        # Create memory for storing received rewards
        received_rewards = {}
        [received_rewards.setdefault(i, []) for i in range(self.bandit.k)]
        self.received_rewards: Dict[int, List] = received_rewards
        self.average_rewards: [] = []

    def take_step(self):
        # Take exploratory step with probability epsilon
        if np.random.normal() <= self.epsilon:
            action_number = np.random.randint(10)
        # Otherwise, take the action with the highest estimated value
        else:
            action_number = np.argmax(self.action_value_estimates)

        # Update probability for that action
        self.received_rewards[action_number].append(self.bandit.get_reward(action_number))
        self.action_value_estimates[action_number] = np.mean(self.received_rewards[action_number])

    def update_average_rewards(self):
        self.average_rewards.append(np.mean(np.concatenate(list(self.received_rewards.values()))))

    def run(self, steps: int):
        for i in range(steps):
            self.take_step()
            self.update_average_rewards()


def run_agent(bandit: Bandit, steps: int) -> []:
    agent = Agent(epsilon=0.1, bandit=bandit)
    agent.run(steps=steps)
    return agent.average_rewards


def main():

    num_steps = 1000
    num_problems = 2000

    rewards = np.empty(shape=(num_problems, num_steps))
    for i in range(num_problems):
        rewards[i] = run_agent(Bandit(k=10), num_steps)

    avg_rewards = np.mean(rewards, axis=0)

    plt.scatter(x=np.arange(start=1, stop=len(avg_rewards) + 1), y=avg_rewards)
    plt.show()


if __name__ == "__main__":
    main()
