import torch
import rlcard

from rlcard.agents.human_agents.leduc_holdem_human_agent import HumanAgent

env = rlcard.make("leduc-holdem")

learning_agent = torch.load(
    "poker_dqn.pth",
    map_location="cpu",
    weights_only=False,
)

human_agent = HumanAgent(num_actions=env.num_actions)
env.set_agents([human_agent, learning_agent])

while True:
    trajectories, payoffs = env.run(is_training=False)

    print("\nGame finished!")
    print("Your payoff:", payoffs[0])
    print("AI payoff:", payoffs[1])

    play_again = input("\nPlay again? Enter y or n: ").lower()

    if play_again != "y":
        print("Thanks for playing!")
        break