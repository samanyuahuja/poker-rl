# Experimental Results

## Matched-Budget Baseline Experiment

The primary experiment compares DQN, NFSP, CFR, and random play in two-player Leduc Hold'em.

DQN and NFSP were trained through self-play. CFR was trained through regret minimization. Random play required no training and was used only during evaluation.

### Experimental Design

Each trained algorithm received:

- 300 seconds of CPU training time per run
- Five independent training seeds: 1, 2, 3, 4, and 5
- The same software environment and hardware
- Model checkpoints recorded from clean Git commit `fe53387`

Each matchup used:

- Five evaluation seeds: 11, 22, 33, 44, and 55
- 2,000 games from each player position
- 20,000 games per training seed
- 100,000 games per matchup
- 600,000 games across all six matchups

Payoff is measured in average chips won per game by the first listed agent. Positive values favor the first agent, while negative values favor its opponent.

For each training seed, results were averaged across evaluation seeds and player positions. The reported 95% confidence intervals were then calculated across the five independently trained models using Student's t distribution.

## Training Budget

The algorithms received equal wall-clock budgets, although their natural training units differ.

| Algorithm | Training unit | Mean completed units | Range |
|---|---|---:|---:|
| DQN | Self-play episodes | 87,239 | 85,951–89,036 |
| NFSP | Self-play episodes | 50,348 | 49,164–51,701 |
| CFR | CFR iterations | 48,515 | 48,155–48,888 |

Episode counts and CFR iterations should not be compared directly. The controlled resource was 300 seconds of training time.

## Matched-Budget Results

| First agent | Opponent | Mean payoff | 95% CI |
|---|---|---:|---:|
| DQN | Random | +0.8839 | ±0.0436 |
| NFSP | Random | +0.5176 | ±0.0144 |
| CFR | Random | +0.7530 | ±0.0445 |
| DQN | NFSP | +0.4726 | ±0.0573 |
| DQN | CFR | -0.0683 | ±0.0704 |
| NFSP | CFR | -0.4474 | ±0.0506 |

## Interpretation

All three trained methods consistently beat random play. DQN extracted the most payoff from the random opponent, followed by CFR and NFSP.

DQN clearly defeated NFSP. Its 95% confidence interval against NFSP remains entirely above zero.

CFR also clearly defeated NFSP. NFSP's payoff against CFR was negative for all five independently trained model pairs.

CFR had a small average advantage over DQN. DQN scored `-0.0683 ± 0.0704` against CFR, but this interval includes zero. The experiment therefore does not establish a statistically reliable winner between DQN and CFR.

The results show that performance against random play and head-to-head strength measure different behavior. DQN exploited random actions most effectively, while CFR remained competitive against DQN and strongly defeated NFSP.

## Reproducibility Files

- `results/matched_baseline_training_manifest.csv` records every training run, completed unit count, elapsed time, and artifact directory.
- `results/matched_baseline_raw.csv` contains all 150 training-seed and evaluation-seed observations.
- `results/matched_baseline_summary.csv` contains the six final matchup estimates and confidence intervals.
- Training checkpoints and generated artifacts are excluded from Git because they can be recreated through the documented commands.

## Earlier Exploratory Results

Earlier experiments evaluated one saved checkpoint per algorithm under unequal training procedures. Those results helped validate the environment and evaluation code but should not be used as the primary comparison between algorithms.

The earlier fixed-checkpoint data remains available in:

- `results/baseline_evaluation.csv`
- `results/cfr_evaluation.csv`
- `results/baseline_summary.png`
- `results/head_to_head_matrix.png`

## Limitations

The experiment uses simplified Leduc Hold'em rather than full Texas Hold'em.

Five training seeds provide an initial estimate of training variability, but additional seeds would produce narrower and more stable confidence intervals.

Wall-clock budgets depend on hardware and background system load. The manifest records completed training units so results can be compared with later replications.

The experiment tests one 300-second training budget. It does not show how the algorithms compare with shorter or longer training.

Exploitability has not yet been measured. Head-to-head payoff and performance against random opponents do not prove that an agent approximates a Nash equilibrium.

The experiment has not yet evaluated adaptive opponent-pool training or its ablations.
