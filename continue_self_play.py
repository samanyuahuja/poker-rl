import torch
import rlcard

from rlcard.utils import reorganize, set_seed, tournament

set_seed(99)

env = rlcard.make("leduc-holdem", config={"seed": 99})

player_0 = torch.load(
    "nfsp_player_0.pth",
    map_location="cpu",
    weights_only=False,
)

player_1 = torch.load(
    "nfsp_player_1.pth",
    map_location="cpu",
    weights_only=False,
)

dqn_agent = torch.load(
    "poker_dqn.pth",
    map_location="cpu",
    weights_only=False,
)

nfsp_agents = [player_0, player_1]
env.set_agents(nfsp_agents)

additional_episodes = 40_000

for episode in range(1, additional_episodes + 1):
    for agent in nfsp_agents:
        agent.sample_episode_policy()

    trajectories, payoffs = env.run(is_training=True)
    trajectories = reorganize(trajectories, payoffs)

    for player_id, agent in enumerate(nfsp_agents):
        for transition in trajectories[player_id]:
            agent.feed(transition)

    if episode % 5000 == 0:
        total_episodes = 10_000 + episode
        print(f"\nReached approximately {total_episodes} total episodes")

env.set_agents([dqn_agent, player_1])
match_one = tournament(env, 5000)

env.set_agents([player_0, dqn_agent])
match_two = tournament(env, 5000)

nfsp_average = (match_one[1] + match_two[0]) / 2
dqn_average = (match_one[0] + match_two[1]) / 2

print("\nTraining and evaluation complete!")
print("NFSP average against DQN:", nfsp_average)
print("DQN average against NFSP:", dqn_average)

torch.save(player_0, "nfsp_player_0_50k.pth")
torch.save(player_1, "nfsp_player_1_50k.pth")

print("Saved the 50k NFSP models")