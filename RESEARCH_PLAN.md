# Research Plan

## Research Question

Under an equal CPU budget in Leduc Hold'em, does uncertainty-aware prioritized historical self-play improve worst-case payoff and approximate exploitability compared with latest-only self-play, uniform historical sampling, and standard Prioritized Fictitious Self-Play?

## Literature Positioning

Historical opponent populations and weakness-based opponent selection are established ideas. This project therefore treats uncertainty-aware sampling as an experimental extension rather than claiming that opponent pools are new.

| Work | Relevant contribution |
|---|---|
| Heinrich, Lanctot, and Silver, 2015 | Introduced sample-based fictitious self-play and evaluated it in imperfect-information poker. |
| Lanctot et al., 2017 | Introduced PSRO, which trains approximate best responses to mixtures drawn from policy populations. |
| Vinyals et al., 2019 | AlphaStar used league training, historical checkpoints, exploiters, and PFSP opponent selection. |
| Balduzzi et al., 2019 | Studied population diversity and strategic cycles in symmetric zero-sum games. |
| Zhang et al., 2026 | Global PSRO used population exploitability to guide policy-population expansion. |

### Sources

- https://proceedings.mlr.press/v37/heinrich15.html
- https://proceedings.neurips.cc/paper_files/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html
- https://www.nature.com/articles/s41586-019-1724-z
- https://proceedings.mlr.press/v97/balduzzi19a.html
- https://proceedings.mlr.press/v306/zhang26ho.html

## Proposed Method

The proposed method maintains a population of historical checkpoints. It prioritizes opponents that appear to exploit the current learner while also assigning additional priority to opponents that have not been evaluated enough.

For opponent `i`:

```text
q_i = estimated probability that opponent i beats the learner

u_i = sqrt(2 * log(N + 1) / (n_i + 1))

score_i = q_i + beta * u_i

```

Where:

- `n_i` is the number of evaluation games against opponent `i`
- `N` is the total number of opponent-evaluation games
- `q_i` counts a loss as 1, a tie as 0.5, and a win as 0
- `beta` controls the uncertainty bonus

Sampling probabilities are:

```text
p_i = (1 - epsilon) * softmax(score_i / temperature)
      + epsilon / pool_size
```

The uniform component ensures every historical opponent retains a nonzero probability of selection.

## Experimental Methods

Four methods will use the same DQN architecture and training code:

1. **Latest-only self-play**

   Train against the newest frozen checkpoint.

2. **Uniform historical self-play**

   Sample every historical checkpoint with equal probability.

3. **Standard PFSP**

   Prioritize historical opponents according to the learner's estimated loss rate.

4. **Uncertainty-aware PFSP**

   Prioritize estimated weaknesses plus an uncertainty bonus.

## Training Protocol

- Environment: two-player Leduc Hold'em
- Training seeds: five independent seeds
- Budget: 300 seconds per method and seed
- Total runs: 20
- Checkpoint interval: approximately every 30 seconds
- Maximum historical pool size: 10
- Opponent-selection overhead counts toward the training budget
- Learner seat alternates during training
- Frozen historical opponents are never updated
- Configuration, environment, progress, pool statistics, and checkpoints are saved for every run

## Evaluation Protocol

Each final policy will be evaluated:

- From both player positions
- Across five evaluation seeds
- Against a fixed reference population
- Against random, DQN, NFSP, and CFR baselines
- Against historical policies from other runs
- With equal games per matchup

Confidence intervals will be calculated across independent training seeds.

## Primary Metrics

1. Worst-case average payoff against the reference population
2. Mean payoff against the reference population
3. Head-to-head payoff against the three alternative training methods
4. Approximate exploitability, if a reliable best-response evaluator is practical

## Secondary Metrics

- Performance against random play
- Opponent-selection distribution
- Number of distinct historical opponents selected
- Training episodes completed
- Checkpoint-to-checkpoint performance
- Wall-clock overhead from opponent evaluation

## Hypotheses

### H1

Historical-opponent training will improve worst-case payoff compared with latest-only self-play.

### H2

Standard PFSP will improve worst-case payoff compared with uniform historical sampling.

### H3

Uncertainty-aware PFSP will improve worst-case payoff compared with standard PFSP by revisiting opponents whose strength estimates remain uncertain.

### H4

The uncertainty bonus will help most early in training and when evaluation samples are limited.

## Required Ablations

- Remove the uncertainty bonus by setting `beta = 0`
- Remove historical checkpoints
- Compare pool sizes of 5, 10, and 20
- Compare at least two uncertainty strengths
- Compare equal wall-clock and equal-episode budgets if time permits

## Claim Boundary

The project will not claim that historical opponent pools, PFSP, or weakness-based sampling are new.

The possible contribution is a controlled, reproducible study of uncertainty-aware opponent prioritization in Leduc Hold'em under a strict compute budget. Any claim of improvement must be supported across independent training seeds and against fixed evaluation opponents.
