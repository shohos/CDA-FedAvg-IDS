# Protocol v3: aggregator study (confirmation)

Status: FROZEN. Binding from the moment its SHA-256 was recorded in
`checkpoints/agg_v3_launch.json`, before any run below starts.

## 1. Question

Which server aggregation rule works best in the non-IID CDA scenario, at low and
high heterogeneity: FedAvg, FedProx, SCAFFOLD or coordinate-wise Median? The
background and the implementation choices are in `AGG_STUDY_PLAN.md`.

## 2. Implementation (frozen at launch)

`final_eval.py --impl repaired --agg <rule> [--mu 0.001]`. This is the
corrected implementation of protocol v2 (causal preprocessing, CUDA RNG isolation,
memory bound L = 1400). With `--agg fedavg` it reproduces the v2 confirmation run
bit for bit. Every source file's SHA-256 is recorded in each result file and in the
launch record.

- **FedProx:** mu = 0.001, chosen on seeds 42-44 by the rule in AGG_STUDY_PLAN.md.
- **SCAFFOLD:** option I control variates; sample-weighted model average; server lr 1.
- **Median:** coordinate-wise, unweighted.

## 3. Runs

- **Seeds:** 800-809, checked unused.
- **Aggregators:** fedavg, fedprox, scaffold, median. One result file per
  aggregator and dataset: `checkpoints/agg/<rule>_<dataset>.json`.
- **Arms:** nonstat (B), rehearsal (B2, the CDA pipeline adapting at every change)
  and cda (C, hybrid detector with quorum 5).
- **Datasets and heterogeneity:** CICIDS2017, UNSW-NB15 and CSE-CIC-IDS2018, each at
  alpha in {0.1, 0.5, 100}. alpha = 0.1 is high heterogeneity and alpha = 100 low.

FedAvg is run again on these seeds rather than borrowed from v2, so every
comparison is paired on identical data.

## 4. Measures

As in protocol v2: the retention score (final pooled macro-F1, the headline),
forward-stream scores, class-by-class forgetting, training steps, memory rows and
detector adaptations.

## 5. Comparisons (paired by seed)

For every arm, dataset and alpha: each rule minus FedAvg, i.e. FedProx - FedAvg,
SCAFFOLD - FedAvg and Median - FedAvg.

**Primary:** those three differences for the rehearsal arm B2 at alpha = 0.1 and
alpha = 100, on each of the three datasets (3 x 2 x 3 = 18 comparisons). Everything
else is secondary and is reported as estimates: alpha = 0.5, arms B and C, the
forward-stream scores, forgetting and cost.

"Best" is decided per heterogeneity level:
- A rule is called better than FedAvg only if its primary difference is clear and
  positive.
- Worse only if it is clear and negative.
- Otherwise there is no clear difference.

## 6. Inference and wording

- Paired 95% t-intervals over the 10 seeds (9 df). **Clear** means the interval
  excludes 0. The primary family is also given with Holm-adjusted p-values.
- No equivalence claims. "Identical" is used only for bit-identical runs.
- The practically relevant difference is 0.02 macro-F1, as in v2.
- Rules are compared with FedAvg, not ranked by point estimates alone. Where no
  difference is clear, the text says so.

## 7. Conduct

- No code change after launch; a crashed run is resumed under the same settings check.
- Jobs may run in parallel (one per aggregator); deviations are appended to the
  launch record.
- Analysis: `analyse_agg.py` (written before any confirmation result is read),
  then the tables and the thesis text.

## 8. Development evidence (consumed seeds 42-44)

Rehearsal arm B2, seeds 42-44 (n = 3), CICIDS2017 and UNSW-NB15, alpha in {0.1, 0.5, 100}
(`checkpoints/agg_dev/`).

**mu for FedProx, by the rule in AGG_STUDY_PLAN.md.** Mean B2 macro-F1 over the 18 cells:

| setting | mean |
|---|---:|
| mu = 0.001 | 0.6045 |
| mu = 0.01 | 0.6059 |
| mu = 0.1 | 0.6065 (best) |
| mu = 1.0 | 0.5760 |
| FedAvg | 0.6057 |
| SCAFFOLD | 0.5295 |
| Median | 0.5702 |

0.001, 0.01 and 0.1 lie within 0.002 of the best, so the rule takes **mu = 0.001**
(`mu_selection.json`).

Development spread of the primary differences, and the difference ten seeds can
detect with 80% power at the 5% level (0.995 x SD):

| contrast | dev mean | paired SD | detectable at n = 10 |
|---|---:|---:|---:|
| cicids a=0.1 fedprox - fedavg | -0.000 | 0.001 | 0.001 |
| cicids a=100 fedprox - fedavg | +0.001 | 0.002 | 0.002 |
| cicids a=0.1 scaffold - fedavg | -0.372 | 0.101 | 0.100 |
| cicids a=100 scaffold - fedavg | -0.009 | 0.002 | 0.002 |
| cicids a=0.1 median - fedavg | -0.188 | 0.047 | 0.047 |
| cicids a=100 median - fedavg | +0.000 | 0.005 | 0.005 |
| unsw a=0.1 fedprox - fedavg | -0.000 | 0.002 | 0.002 |
| unsw a=100 fedprox - fedavg | -0.001 | 0.001 | 0.001 |
| unsw a=0.1 scaffold - fedavg | -0.059 | 0.076 | 0.076 |
| unsw a=100 scaffold - fedavg | -0.010 | 0.004 | 0.004 |
| unsw a=0.1 median - fedavg | -0.051 | 0.052 | 0.052 |
| unsw a=100 median - fedavg | +0.011 | 0.021 | 0.021 |

These numbers set mu and describe detectability only; they are not results.
At alpha = 0.1, SCAFFOLD and Median vary a lot between seeds, so there only
differences of about 0.05 to 0.10 can be detected; elsewhere 0.001 to 0.02.
