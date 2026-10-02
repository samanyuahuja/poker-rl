import torch
import rlcard

from rlcard.utils import tournament

env = rlcard.make("leduc-holdem", config={"seed": 200})

dqn_agent = torch.load(
    "poker_dqn.pth",
    map_location="cpu",
    weights_only=False,
)

nfsp_player_0 = torch.load(
    "nfsp_player_0.pth",
    map_location="cpu",
    weights_only=False,
)

nfsp_player_1 = torch.load(
    "nfsp_player_1.pth",
    map_location="cpu",
    weights_only=False,
)

# DQN occupies seat 0; NFSP occupies seat 1.
env.set_agents([dqn_agent, nfsp_player_1])
first_match = tournament(env, 5000)

# Swap the model positions.
env.set_agents([nfsp_player_0, dqn_agent])
second_match = tournament(env, 5000)

nfsp_average = (first_match[1] + second_match[0]) / 2
dqn_average = (first_match[0] + second_match[1]) / 2

print("Match 1 — DQN seat 0:", first_match[0])
print("Match 1 — NFSP seat 1:", first_match[1])

print("Match 2 — NFSP seat 0:", second_match[0])
print("Match 2 — DQN seat 1:", second_match[1])

print("\nNFSP average against DQN:", nfsp_average)
print("DQN average against NFSP:", dqn_average)