# Experimental Results

## Baseline Evaluation

The saved DQN and 50,000-episode NFSP checkpoints were evaluated in Leduc Hold'em across five evaluation seeds.

Each matchup used:

- 5 evaluation seeds
- 2,000 games per seed
- 10,000 games per seat configuration
- Average payoff measured in chips per game

## Results

| Evaluated agent and matchup | Mean payoff | 95% CI |
|---|---:|---:|
| DQN vs random, seat 0 | +1.3720 | ±0.0544 |
| DQN vs random, seat 1 | +1.3337 | ±0.0249 |
| NFSP vs random, seat 0 | +0.5208 | ±0.0087 |
| NFSP vs random, seat 1 | +0.5065 | ±0.0401 |
| DQN vs NFSP, DQN in seat 0 | +0.6836 | ±0.1323 |
| DQN vs NFSP, DQN in seat 1 | +0.9867 | ±0.1086 |
| DQN vs NFSP, both seats combined | +0.8352 | ±0.0304 |

## CFR Baseline

CFR was trained for 1,000 iterations and evaluated across five seeds, with 2,000 games per seed and player position.

| CFR matchup, both seats combined | Mean payoff | 95% CI |
|---|---:|---:|
| CFR vs random | +0.8070 | ±0.0162 |
| CFR vs DQN | +0.8807 | ±0.0507 |
| CFR vs NFSP | +0.5514 | ±0.0753 |

CFR earned less against random play than DQN, but defeated both learned agents. This suggests that the DQN learned to exploit weak random behavior while remaining vulnerable to a more balanced game-theoretic strategy.

## Interpretation

Both trained methods consistently beat random play. The DQN extracted more value from the random opponent and defeated the NFSP checkpoints from both seats.

The result does not prove that DQN is generally superior to NFSP. The algorithms used different training procedures, and only one trained checkpoint per method was evaluated.

## Limitations

The reported intervals measure variation across evaluation seeds using fixed trained models. They do not capture variation caused by retraining each algorithm from different initialization seeds.

A stronger comparison requires:

- Multiple independently trained models per algorithm
- Equal training budgets
- A CFR baseline
- Exploitability measurements
- Training curves and checkpoint evaluation
- Statistical comparisons across training runs

The raw evaluation data is stored in `results/baseline_evaluation.csv`.