# Poker Reinforcement Learning

A reinforcement-learning project for studying decision-making in imperfect-information games using Leduc Hold'em.

The project compares a Deep Q-Network (DQN) trained against a random opponent with Neural Fictitious Self-Play (NFSP) agents trained against each other.

## Current Results

| Experiment | Average payoff |
|---|---:|
| DQN vs random, first seat | +1.3128 |
| DQN vs random, second seat | +1.3568 |
| DQN vs random, both seats | +1.3348 |
| NFSP player 0 vs random | +0.6480 |
| NFSP player 1 vs random | +0.7002 |
| DQN vs initial NFSP | approximately +0.55 |

Positive payoff means the listed agent won chips on average.

These are fixed-checkpoint evaluation results. The intervals measure evaluation-seed variation, not retraining variance, and exploitability has not yet been measured.

![Baseline evaluation results](results/baseline_summary.png)

See [RESULTS.md](RESULTS.md) for the complete methodology, numerical results, and limitations.

### Head-to-Head Comparison

![Head-to-head payoff matrix](results/head_to_head_matrix.png)

Positive cells favor the row agent. CFR defeated DQN and NFSP despite earning less than DQN against random play, demonstrating the difference between exploiting weak opponents and learning a robust strategy.

## Features

- Leduc Hold'em simulation through RLCard
- DQN training against random play
- NFSP self-play training
- Evaluation from both player positions
- Head-to-head model comparison
- Terminal interface for playing against a trained model
- Local model checkpoints excluded from Git

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Reproducible Experiment CLI

The primary interface is the `poker_rl` command-line package.

Train one agent with a fixed wall-clock budget:

```bash
python -m poker_rl train \
  --algorithm dqn \
  --seed 1 \
  --budget-seconds 300
```

Valid algorithms are `dqn`, `nfsp`, and `cfr`.

Train multiple baselines under the same time budget:

```bash
python -m poker_rl suite \
  --algorithms dqn nfsp cfr \
  --seeds 1 2 3 4 5 \
  --budget-seconds 300 \
  --output artifacts/baseline_suite.csv
```

Evaluate two trained agents from both player positions:

```bash
python -m poker_rl evaluate \
  --agent-a artifacts/baseline-dqn-seed1-budget300s \
  --agent-b random \
  --seeds 11 22 33 44 55 \
  --games-per-seat 10000 \
  --output results/dqn_vs_random.csv
```
Run the complete matched-budget baseline tournament:

```bash
python -m poker_rl benchmark \
  --manifest artifacts/baseline_suite.csv \
  --evaluation-seeds 11 22 33 44 55 \
  --games-per-seat 2000 \
  --raw-output results/matched_baseline_raw.csv \
  --summary-output results/matched_baseline_summary.csv
```

The benchmark evaluates DQN, NFSP, CFR, and random play through six matchups. Every trained method receives the same wall-clock training budget and is evaluated from both player positions.

The reported 95% confidence intervals measure variation across independently trained agents. Evaluation seeds and both player positions are averaged within each training seed before calculating the interval.

Use `random` as either agent to select the random-policy baseline.

Each training run saves:

- `config.json`: requested algorithm, seed, and budget
- `environment.json`: Python, platform, and Git metadata
- `summary.json`: completed units and elapsed training time
- `training_progress.csv`: progress measurements during training
- Model checkpoints needed for later evaluation

Generated run artifacts and model checkpoints are excluded from Git because they can be recreated from the recorded configuration.

The comparison budget is wall-clock training time. DQN and NFSP report self-play episodes, while CFR reports iterations, so those unit counts should not be compared directly.

## Legacy Commands

The original standalone scripts remain available for reproducing the early experiments:

```bash
python main.py
python evaluate.py
python train_self_play.py
python continue_self_play.py
python compare_models.py
python play.py
```

## Research Roadmap

- Create a reproducible experiment runner
- Add CFR as a game-theoretic baseline
- Run experiments across multiple random seeds
- Calculate confidence intervals and learning curves
- Measure exploitability
- Test adaptive opponent-pool training
- Conduct ablation studies
- Produce a paper-style report

## Limitations

The current agents play simplified Leduc Hold'em rather than full Texas Hold'em. Performance against random opponents does not establish optimal or generally strong poker play. The current results should be treated as an early experimental baseline.