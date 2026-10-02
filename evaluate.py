import torch
import rlcard

from rlcard.agents import RandomAgent
from rlcard.utils import tournament

env = rlcard.make("leduc-holdem", config={"seed": 100})

learning_agent = torch.load(
    "poker_dqn.pth",
    map_location="cpu",
    weights_only=False,
)

random_agent = RandomAgent(num_actions=env.num_actions)

env.set_agents([learning_agent, random_agent])
first_seat_results = tournament(env, 5000)

env.set_agents([random_agent, learning_agent])
second_seat_results = tournament(env, 5000)

first_seat_score = first_seat_results[0]
second_seat_score = second_seat_results[1]
combined_score = (first_seat_score + second_seat_score) / 2

print("DQN score in first seat:", first_seat_score)
print("DQN score in second seat:", second_seat_score)
print("Average across both seats:", combined_score)