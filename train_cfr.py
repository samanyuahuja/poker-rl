from pathlib import Path

import rlcard

from rlcard.agents import CFRAgent, RandomAgent
from rlcard.utils import set_seed, tournament

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "checkpoints" / "cfr"

set_seed(42)

env = rlcard.make(
    "leduc-holdem",
    config={
        "seed": 42,
        "allow_step_back": True,
    },
)

cfr_agent = CFRAgent(
    env,
    model_path=str(MODEL_DIR),
)

number_of_iterations = 1000

for iteration in range(1, number_of_iterations + 1):
    cfr_agent.train()

    if iteration % 100 == 0:
        print(
            f"Finished CFR iteration "
            f"{iteration}/{number_of_iterations}"
        )

cfr_agent.save()
print(f"Saved CFR model to {MODEL_DIR}")

random_agent = RandomAgent(num_actions=env.num_actions)

env.set_agents([cfr_agent, random_agent])
seat_0_payoff = tournament(env, 5000)[0]

env.set_agents([random_agent, cfr_agent])
seat_1_payoff = tournament(env, 5000)[1]

combined_payoff = (seat_0_payoff + seat_1_payoff) / 2

print("\nCFR evaluation complete!")
print("CFR vs random, seat 0:", seat_0_payoff)
print("CFR vs random, seat 1:", seat_1_payoff)
print("CFR vs random, both seats:", combined_payoff)