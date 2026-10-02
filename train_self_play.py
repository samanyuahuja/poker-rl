import torch
import rlcard

from rlcard.agents import NFSPAgent, RandomAgent
from rlcard.utils import reorganize, set_seed, tournament

set_seed(42)

env = rlcard.make("leduc-holdem", config={"seed": 42})
device = torch.device("cpu")


def create_nfsp_agent():
    return NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[64, 64],
        q_mlp_layers=[64, 64],
        device=device,
    )


player_0 = create_nfsp_agent()
player_1 = create_nfsp_agent()

agents = [player_0, player_1]
env.set_agents(agents)

number_of_episodes = 10_000

for episode in range(1, number_of_episodes + 1):
    for agent in agents:
        agent.sample_episode_policy()

    trajectories, payoffs = env.run(is_training=True)
    trajectories = reorganize(trajectories, payoffs)

    for player_id, agent in enumerate(agents):
        for transition in trajectories[player_id]:
            agent.feed(transition)

    if episode % 1000 == 0:
        print(f"\nFinished episode {episode}/{number_of_episodes}")

random_agent = RandomAgent(num_actions=env.num_actions)

env.set_agents([player_0, random_agent])
player_0_score = tournament(env, 3000)[0]

env.set_agents([random_agent, player_1])
player_1_score = tournament(env, 3000)[1]

print("\nSelf-play training complete!")
print("Player 0 model vs random:", player_0_score)
print("Player 1 model vs random:", player_1_score)

torch.save(player_0, "nfsp_player_0.pth")
torch.save(player_1, "nfsp_player_1.pth")

print("Saved nfsp_player_0.pth")
print("Saved nfsp_player_1.pth")