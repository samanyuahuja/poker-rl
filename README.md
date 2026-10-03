# Poker Reinforcement Learning

A reproducible reinforcement-learning research project for studying decision-making in imperfect-information games using two-player Leduc Hold'em.

The project trains and compares Deep Q-Network (DQN), Neural Fictitious Self-Play (NFSP), Counterfactual Regret Minimization (CFR), and random-policy agents.

## Current Results

The primary experiment gave DQN, NFSP, and CFR the same 300-second training budget across five independent training seeds. Each matchup used five evaluation seeds, both player positions, and 100,000 total games.

| First agent | Opponent | Mean payoff | 95% CI |
|---|---|---:|---:|
| DQN | Random | +0.8839 | ±0.0436 |
| NFSP | Random | +0.5176 | ±0.0144 |
| CFR | Random | +0.7530 | ±0.0445 |
| DQN | NFSP | +0.4726 | ±0.0573 |
| DQN | CFR | -0.0683 | ±0.0704 |
| NFSP | CFR | -0.4474 | ±0.0506 |

Positive payoff favors the first listed agent.

DQN and CFR both clearly defeated NFSP. DQN extracted the most payoff from random play. CFR had a small average advantage over DQN, but the DQN-versus-CFR confidence interval includes zero, so the experiment does not establish a reliable winner between them.

The experiment covers 15 independently trained models and 600,000 evaluation games. Confidence intervals measure variation across training seeds rather than repeated evaluation of one fixed checkpoint.

See [RESULTS.md](RESULTS.md) for the complete methodology, interpretation, raw-data references, and limitations.

## Features

- Two-player Leduc Hold'em simulation through RLCard
- DQN and NFSP self-play training
- CFR game-theoretic baseline
- Equal wall-clock training budgets across algorithms
- Multi-seed training and evaluation suites
- Evaluation from both player positions
- Head-to-head baseline tournament
- 95% confidence intervals across independent training runs
- Raw results, summaries, and training manifests saved as CSV
- Terminal interface for playing against a trained model
- Generated model checkpoints excluded from Git

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

- [x] GitHub setup and project hygiene
- [x] Reproducible training and evaluation commands
- [x] Equal-budget DQN, NFSP, CFR, and random baselines
- [x] Research question and literature review
- [ ] Adaptive opponent-pool method
- [ ] Rigorous proposed-method experiments
- [ ] Ablation studies
- [ ] Final analysis and visualizations
- [ ] Paper-style report
- [ ] Resume-ready release

## Limitations

The current agents play simplified Leduc Hold'em rather than full Texas Hold'em. Head-to-head payoff and performance against random opponents do not establish optimal play. Exploitability, adaptive opponent-pool training, and ablation studies remain future stages. See [RESULTS.md](RESULTS.md) for detailed experimental limitations. See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the literature positioning, proposed uncertainty-aware method, hypotheses, and experimental protocol.
