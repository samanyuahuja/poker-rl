# Uncertainty-Aware Historical Opponent Sampling Under a Fixed Compute Budget in Leduc Hold'em

**Samanyu Ahuja**

## Abstract

Self-play agents can become narrowly adapted to recent opponents and lose strategies that worked against older policies. A historical opponent pool preserves earlier checkpoints, but it still needs a rule for choosing the next opponent. This paper tests an uncertainty-aware sampler that combines the learner's estimated weakness against each opponent with an exploration bonus for opponents evaluated fewer times. We compare it with latest-only self-play, uniform historical sampling, and Prioritized Fictitious Self-Play (PFSP) in two-player Leduc Hold'em. All four strategies use the same DQN architecture, five independent training seeds, and a 300-second CPU budget per seed. Their final policies are evaluated from both seats against random, DQN, NFSP, and CFR reference agents, as well as against one another. Uniform sampling had the highest mean-reference payoff at `+0.3226 +/- 0.0929`. PFSP had the strongest worst-case point estimate at `-0.1749 +/- 0.1551`. The uncertainty-aware method reached `+0.2417 +/- 0.0874` mean-reference payoff and `-0.2212 +/- 0.1723` worst-case payoff. Paired comparisons found no improvement over PFSP, and the uncertainty-strength and pool-size ablations did not identify a clear winner. Under this compute budget, the extra uncertainty bonus did not improve the measured outcomes.

## 1. Introduction

Poker requires players to act without seeing an opponent's private cards. Leduc Hold'em retains this imperfect-information structure in a game small enough to support repeated training and evaluation.

Self-play generates its own training data, but the opponent changes throughout training. Using only the newest checkpoint can push the learner toward a narrow response to the most recent policy. Keeping older checkpoints broadens the available opponents, though the result depends on how those opponents are sampled and how much time is spent evaluating them.

The research question is:

> Under an equal CPU budget in Leduc Hold'em, does uncertainty-aware historical opponent sampling improve mean or worst-case payoff compared with latest-only self-play, uniform historical sampling, and standard PFSP?

The tested sampler gives more weight to opponents that currently beat the learner and to opponents whose estimates come from fewer evaluation games. Historical populations and weakness-based sampling are established methods. The question here is narrower: does adding an uncertainty bonus help when every strategy receives the same wall-clock budget?

The experiment found no advantage for the uncertainty-aware sampler over PFSP. Its default configuration also trailed uniform sampling in the paired mean-reference comparison. The added selection rule therefore did not justify its complexity under the tested conditions.

## 2. Related Work

Counterfactual Regret Minimization (CFR) minimizes counterfactual regret in extensive-form games and can approach a Nash equilibrium in two-player zero-sum settings. Zinkevich et al. evaluated it on large poker abstractions [1].

Fictitious Self-Play (FSP) adapted fictitious play to sample-based learning in extensive-form games and tested it in imperfect-information poker [2]. Neural Fictitious Self-Play (NFSP) later combined that approach with deep reinforcement learning and learned approximate equilibrium strategies in Leduc poker [3]. This project includes NFSP and CFR as reference agents because both were designed for imperfect-information games.

Policy-Space Response Oracles (PSRO) trains approximate best responses to meta-strategies over a policy population and unifies several earlier game-solving procedures [4]. Balduzzi et al. later analyzed strategic cycles and population diversity in non-transitive games, where policies cannot always be arranged in a single ranking from weak to strong [5].

AlphaStar trained a league containing current agents, exploiters, and historical policies. Prioritized opponent selection was one part of that system [6]. Historical checkpoints and PFSP-style selection therefore predate this project. Global PSRO provides a more recent population method that guides expansion with population-level quality and exploitability estimates [7].

The code uses RLCard, which provides the card-game environment and implementations of DQN, NFSP, and CFR [8]. The uncertainty term is an experimental extension to established opponent-pool methods.

## 3. Methods

### 3.1 Environment and learner

All runs use RLCard's two-player Leduc Hold'em environment. The pool learner is a CPU-based DQN with two 64-unit hidden layers. Training begins by copying the learner into a frozen greedy opponent. Only the learner is updated, and its seat alternates after every episode.

The trainer adds frozen checkpoints on a wall-clock schedule and removes the oldest one when the pool is full. Periodic opponent evaluation, checkpoint creation, and opponent selection all consume time from the 300-second budget. The code records a separate final evaluation after training ends.

### 3.2 Opponent-selection strategies

Four selection rules are compared:

1. **Latest:** always train against the newest checkpoint.
2. **Uniform:** assign equal probability to every active checkpoint.
3. **PFSP:** assign priority proportional to the square of the estimated learner loss rate, then mix the distribution with uniform sampling.
4. **Uncertainty:** combine estimated learner loss rate with an uncertainty bonus, apply a softmax, and mix the result with uniform sampling.

For historical opponent `i`, let `q_i` be its estimated score against the learner, where a learner loss contributes 1, a tie contributes 0.5, and a learner win contributes 0. Let `n_i` be the number of evaluation games against that opponent and let `N` be the total evaluation count across the active pool. The uncertainty score implemented in the project is

```text
u_i = sqrt(2 * log(N + 2) / (n_i + 1))
score_i = q_i + beta * u_i
```

The sampling distribution is

```text
p_i = (1 - epsilon) * softmax(score_i / temperature)
      + epsilon / pool_size.
```

The primary uncertainty configuration uses `beta=1`, `epsilon=0.1`, and `temperature=1`. The uniform component gives every active opponent nonzero sampling probability.

### 3.3 Training protocol

Each primary strategy is trained with:

- five independent training seeds: 1, 2, 3, 4, and 5;
- 300 seconds of CPU training per seed;
- a 30-second checkpoint interval;
- a maximum active pool size of 10;
- 100 seat-balanced evaluation games per active opponent at each scheduled evaluation;
- identical DQN creation and update code.

This creates 20 primary training runs. The controlled resource is wall-clock time because DQN, NFSP, and CFR have different natural training units. Every run stores its configuration, environment metadata, progress, pool statistics, completion summary, final policies, and historical checkpoints.

### 3.4 Evaluation protocol

Each final policy plays four fixed references: random, matched-budget DQN, matched-budget NFSP, and matched-budget CFR. The evaluation also covers every pair of pool strategies.

For each matchup, the agents play 2,000 games from each seat under five evaluation seeds. Trained opponents use matching training seeds. The analysis first averages the two seats within an evaluation seed, then averages evaluation seeds within each training seed. The five independent training-seed values determine the mean and 95% Student's t confidence interval.

Two aggregate reference metrics are reported:

- **Mean-reference payoff:** the average payoff against random, DQN, NFSP, and CFR within each training seed.
- **Worst-case reference payoff:** the minimum of those four reference payoffs within each training seed.

Both metrics summarize performance against this fixed reference set. Neither is an exact exploitability estimate.

### 3.5 Controlled ablations

The ablation study trains five additional configurations across the same five seeds and 300-second budgets:

- uncertainty strength `beta=0` and `beta=2`, compared with the primary `beta=1` control;
- maximum pool sizes 5, 10, and 20;
- latest-only training as the comparison without historical sampling.

Pool-size runs use a 10-second checkpoint interval so the larger pools can fill. Every ablation final policy is evaluated against the four fixed references with the same five evaluation seeds and 2,000 games per seat.

## 4. Results

### 4.1 Matched-budget baselines

Before the pool experiment, DQN, NFSP, and CFR each received 300 seconds of training across five seeds. All three beat random play. DQN beat NFSP by `+0.4726 +/- 0.0573`. Its payoff against CFR was `-0.0683 +/- 0.0704`, an interval that includes zero. DQN extracted more payoff from random play, but that result did not make it clearly stronger than CFR.

### 4.2 Primary opponent-pool results

The primary study contains 20 training runs and 2.2 million evaluation games.

| Strategy | Mean-reference payoff | Worst-case reference payoff |
|---|---:|---:|
| Latest | +0.0935 +/- 0.2197 | -0.4676 +/- 0.2873 |
| Uniform | **+0.3226 +/- 0.0929** | -0.3108 +/- 0.3583 |
| PFSP | +0.3180 +/- 0.0766 | **-0.1749 +/- 0.1551** |
| Uncertainty | +0.2417 +/- 0.0874 | -0.2212 +/- 0.1723 |

![Aggregate primary opponent-pool results](results/pool_strategy_aggregate.png)

Uniform sampling recorded the highest mean-reference point estimate, while PFSP recorded the strongest worst-case point estimate. In the paired seed-level analysis, uniform exceeded latest on mean-reference payoff by `+0.2291 +/- 0.1984` and uncertainty by `+0.0809 +/- 0.0640`.

The proposed uncertainty method did not improve on PFSP:

| Paired difference | Estimate |
|---|---:|
| Uncertainty minus PFSP, mean-reference payoff | -0.0763 +/- 0.1457 |
| Uncertainty minus PFSP, worst-case payoff | -0.0463 +/- 0.2135 |

Both intervals include zero, although their point estimates favor PFSP. The six direct head-to-head comparisons are also inconclusive because every interval includes zero.

![Primary head-to-head payoff matrix](results/pool_head_to_head_heatmap.png)

The hypotheses receive the following outcomes:

- **H1, historical sampling improves worst-case payoff over latest-only:** not established. Historical methods had stronger point estimates, but paired worst-case intervals included zero.
- **H2, PFSP improves worst-case payoff over uniform:** not established. PFSP had the stronger point estimate, but the paired difference was inconclusive.
- **H3, uncertainty-aware sampling improves over PFSP:** not supported. Neither the mean nor worst-case comparison showed an advantage.

### 4.3 Ablation results

The ablation study adds 25 training runs and 2.0 million evaluation games.

| Configuration | Mean-reference payoff | Worst-case reference payoff |
|---|---:|---:|
| Beta 0 | +0.3490 +/- 0.0411 | -0.2104 +/- 0.0635 |
| Beta 1 | +0.2417 +/- 0.0874 | -0.2212 +/- 0.1723 |
| Beta 2 | +0.2311 +/- 0.1161 | -0.2295 +/- 0.1149 |
| Pool 5 | +0.2777 +/- 0.1224 | -0.1196 +/- 0.1110 |
| Pool 10 | +0.3051 +/- 0.1747 | -0.1138 +/- 0.1038 |
| Pool 20 | +0.2830 +/- 0.1309 | -0.1154 +/- 0.0972 |

![Controlled ablation results](results/ablation_aggregate.png)

Beta 0 recorded the highest mean-reference point estimate among the uncertainty-strength settings. Pool 10 recorded the highest mean and worst-case point estimates among the pool-size settings. All 14 paired ablation intervals include zero, so these runs do not establish an effect from uncertainty strength, pool size, or latest-only versus default uncertainty-aware training.

The ablations do not support **H4** under the tested settings. Increasing the uncertainty strength brought no clear benefit. The design did not directly isolate early training or reduce the opponent-evaluation budget, so it cannot rule out an effect in either condition.

## 5. Discussion

Under the tested budget, the simpler samplers were at least as effective as the proposed method. Uniform sampling performed best on average against the fixed references, and PFSP had the strongest worst-case point estimate. The uncertainty term improved neither metric.

Budget allocation may explain part of the result. Evaluating each active opponent leaves less time for learner updates. The bonus may then spend some of the remaining episodes on opponents with fewer observations instead of those causing the largest measured losses. The implementation allows this mechanism, but the experiment did not measure its causal effect.

Uniform sampling keeps exposure spread across the historical pool, which may limit specialization. PFSP directs more updates toward policies that beat the learner, consistent with its stronger worst-case point estimate. Latest-only training uses a narrow moving target and produced the most variable diagnostic curve. These explanations remain hypotheses.

![Training payoff diagnostics](results/training_learning_curves.png)

Each training curve reports payoff against whichever opponents that strategy sampled at the time. Since those opponent distributions change and differ across strategies, the curves are diagnostic and are not directly comparable to fixed-policy evaluation.

The negative result sets a useful boundary on the method. With five 300-second CPU runs per configuration, the results provide no reason to prefer the uncertainty bonus, even though some unpaired point estimates look different. The paired analysis also shows how much training variation remains with five seeds.

## 6. Limitations

The evidence is limited in several ways:

1. Leduc Hold'em is much smaller than full Texas Hold'em, so the result may not transfer to larger games.
2. Five training seeds give a useful first estimate of variability but leave several confidence intervals wide.
3. Wall-clock equality is hardware-dependent and allows completed episode counts to vary across runs.
4. Only one primary training duration, 300 seconds, is tested.
5. The fixed reference population is limited to random, DQN, NFSP, and CFR agents produced by this project.
6. Mean and worst-case reference payoff are not exploitability. No exact or approximate best-response exploitability evaluator was completed.
7. The ablations vary uncertainty strength and pool size, but they do not isolate opponent-evaluation frequency, evaluation cost, selection entropy, or forgetting against evicted checkpoints.
8. Multiple exploratory comparisons are reported without a family-wise error correction. Their confidence intervals should be read as estimation aids rather than definitive discovery tests.

## 7. Conclusion

With matched CPU budgets in Leduc Hold'em, uniform historical sampling produced the highest mean-reference payoff and PFSP produced the strongest worst-case point estimate. The uncertainty-aware extension did not improve on PFSP. Changing its bonus strength and pool size also failed to produce a clear winner.

This result applies to the tested implementation and budget, not to every possible uncertainty-aware sampler. A stronger follow-up would use more training seeds, separate evaluation overhead from the learner-update budget, track selection entropy and forgetting, vary evaluation frequency, and include a reliable exploitability or best-response estimate.

## 8. Reproducibility

The repository provides commands for training, evaluation, benchmarking, paired analysis, ablations, and plotting. It includes run manifests, raw and summarized results, tests, and generated figures. Model checkpoints and generated training artifacts are excluded from Git; their configurations are recorded so the runs can be recreated.

Primary evidence files include:

- `results/matched_baseline_raw.csv`
- `results/pool_benchmark_raw.csv`
- `results/pool_benchmark_paired.csv`
- `results/ablation_benchmark_raw.csv`
- `results/ablation_benchmark_paired.csv`
- `plot_research_results.py`

Repository: [github.com/samanyuahuja/poker-rl](https://github.com/samanyuahuja/poker-rl)

## References

1. Zinkevich, M., Johanson, M., Bowling, M., and Piccione, C. (2007). [Regret Minimization in Games with Incomplete Information](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html). *Advances in Neural Information Processing Systems 20*.
2. Heinrich, J., Lanctot, M., and Silver, D. (2015). [Fictitious Self-Play in Extensive-Form Games](https://proceedings.mlr.press/v37/heinrich15.html). *Proceedings of Machine Learning Research 37*, 805-813.
3. Heinrich, J., and Silver, D. (2016). [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121). arXiv:1603.01121.
4. Lanctot, M., Zambaldi, V., Gruslys, A., Lazaridou, A., Perolat, J., Silver, D., and Graepel, T. (2017). [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://proceedings.neurips.cc/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html). *Advances in Neural Information Processing Systems 30*.
5. Balduzzi, D., Garnelo, M., Bachrach, Y., Czarnecki, W., Perolat, J., Jaderberg, M., and Graepel, T. (2019). [Open-Ended Learning in Symmetric Zero-Sum Games](https://proceedings.mlr.press/v97/balduzzi19a.html). *Proceedings of Machine Learning Research 97*, 434-443.
6. Vinyals, O. et al. (2019). [Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning](https://www.nature.com/articles/s41586-019-1724-z). *Nature 575*, 350-354.
7. Zhang, J., Yang, F., Wang, J., Wang, C., and Zhang, X. (2026). [Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](https://proceedings.mlr.press/v306/zhang26ho.html). *Proceedings of Machine Learning Research 306*, 159859-159886.
8. Zha, D. et al. (2020). [RLCard: A Platform for Reinforcement Learning in Card Games](https://ijcai.org/proceedings/2020/764). *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence*, 5264-5266.
