import torch
import rlcard

from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import reorganize, set_seed, tournament

set_seed(42)

env = rlcard.make("leduc-holdem", config={"seed": 42})
device = torch.device("cpu")

learning_agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[64, 64],
    device=device,
)

random_agent = RandomAgent(num_actions=env.num_actions)
env.set_agents([learning_agent, random_agent])

number_of_episodes = 5000

for episode in range(1, number_of_episodes + 1):
    trajectories, payoffs = env.run(is_training=True)
    trajectories = reorganize(trajectories, payoffs)

    for transition in trajectories[0]:
        learning_agent.feed(transition)

    if episode % 500 == 0:
        print(f"Finished episode {episode}/{number_of_episodes}")

average_payoffs = tournament(env, 1000)

print("Training complete!")
print("DQN average payoff:", average_payoffs[0])
print("Random agent average payoff:", average_payoffs[1])

torch.save(learning_agent, "poker_dqn.pth")
print("Model saved as poker_dqn.pth")