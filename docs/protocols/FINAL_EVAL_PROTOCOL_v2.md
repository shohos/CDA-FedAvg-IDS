# Final evaluation protocol v2 (after the 2026-09-26 audit repairs)

Status: FROZEN. Binding from the moment its SHA-256 was recorded in
`checkpoints/final_eval_v2_launch.json`, before any run below starts.

## 1. Why a second protocol

The independent audit of 2026-09-26 found implementation problems that bear on
the original final evaluation (`FINAL_EVAL_PROTOCOL.md`, seeds 47-56):
- preprocessing fitted on future data;
- a CUDA random-state leak between the detector and the classifier;
- an unbounded memory;
- undisclosed FL-IIDS equation choices.

`AUDIT_REPAIR_PLAN.md` records the repairs and their checks. The original results
stay as they are (`checkpoints/final_*.json`). This protocol fixes, before any
run:
- how the repaired implementation is evaluated;
- which comparisons are made;
- how they are judged.

Development used consumed seeds only (41-46, `checkpoints/audit_repair_dev/`).

## 2. Implementation (frozen at launch)

`final_eval.py --impl repaired --fliids-lr 0.02`, which sets:
- causal preprocessing: fill values, category codes and scaler from the first concept's training rows only;
- CUDA RNG isolation;
- memory bound L = 1400 with class-balanced random eviction;
- FL-IIDS rows beyond the quota discarded;
- FL-IIDS teacher probabilities over the old classes only.

The Eq. 8 target runs under both readings (raw and normalised), as separate arms.

All other settings are as in protocol v1: momentum SGD, lr 0.01, 10 local epochs, batch 100, clip 10, LayerNorm, 5 rounds per concept, 10 clients, corrected CICIDS2017 labels, hybrid detector with OR fusion. The SHA-256 of every source file is recorded in each result file and in the launch record; a result file written with different code is refused on resume.

## 3. Arms

| Key | Arm |
|---|---|
| iid_stream | A2, training-matched IID reference |
| nonstat | B, time order, no memory |
| rehearsal | B2, memory, always adapt |
| cda | C5, detector, quorum 5 |
| cda_q1 | C1, detector, any one client |
| fliids | F, FL-IIDS-derived, Eq. 9 loss, raw Eq. 8 target |
| fliids_norm | Fn, the same, normalised Eq. 8 target |
| fliids_ce | F0, the same memory and schedule, cross-entropy weight 1 |
| fliids_ce_half | F0h, cross-entropy weight 0.5 (Eq. 9's scale) |
| fliids_gb | Fgb, 0.5 x gradient-balance loss only |
| fliids_ce_ls | Fls, 0.5 CE + 0.5 label smoothing, raw target |
| fliids_ce_ls_norm | Flsn, the same, normalised target |

## 4. Runs

**Datasets and heterogeneity.** CICIDS2017 and UNSW-NB15 at alpha in {0.1, 0.5, 1.0, 100}; CSE-CIC-IDS2018 at alpha = 0.5 (daily concepts). All twelve arms in every cell.

**R1, paired diagnostic, seeds 47-56.** These seeds were consumed by the original final evaluation. R1 against `final_*.json`, paired by seed, is the before/after accounting of the repairs. No new claim rests on R1 alone.

**R2, confirmation, seeds 600-609.** Checked unused before launch (used: 0-10, 41-56, 100-104, 200-209). Every claim the thesis makes about the repaired implementation rests on R2.

## 5. Measures

1. **Retention score:** macro-F1 on the pooled held-out test set after the last concept.
2. **Forward-stream scores**, on each arriving concept's held-out rows, just before the model trains on it, averaged over concepts 2..T:
   - macro-F1 over the classes present;
   - detection: the share of attack rows flagged as any attack;
   - false alarm: the share of benign rows flagged.
3. **Class-by-class forgetting**, as in protocol v1: best recall from the introducing concept up to the second-to-last concept, minus final recall, floored at 0. Also the counts of classes learned and learned-then-lost.
4. **Cost:** classifier gradient steps. Detector autoencoder training, teacher passes and herding are not counted, so this is not total compute.
5. **Memory:** peak rows stored per client (B2, C: rehearsal memory; F arms: exemplar rows kept).
6. **Detector:** number of concepts at which the federation adapted, per run.

## 6. Comparisons (paired by seed, at every alpha where both arms exist)

| Family | Differences |
|---|---|
| Memory | B2 - B |
| Detector | C5 - B2, C1 - B2 |
| Time order | B2 - A2 |
| FL-IIDS vs ours | F - B2, Fn - B2 |
| FL-IIDS losses | F - F0, Fn - F0 |
| Loss decomposition | F0h - F0 (step size), Fgb - F0h (gradient balance), Fls - F0h and Flsn - F0h (label smoothing), F - Fgb and Fn - Fgb (label smoothing on top of gradient balance) |
| Cost of heterogeneity | score(100) - score(a) for a in {0.1, 0.5} for B, B2, C5, F, Fn, F0; and cost(F) - cost(B2), cost(Fn) - cost(B2) |
| Forward-stream | B2 - B, C5 - B2, F - B2 on forward macro-F1 and detection |

**Primary contrasts.** The thesis states conclusions only from these; everything else is secondary and reported as estimates:

| # | Contrast | Where |
|---|---|---|
| P1 | B2 - B | alpha = 0.5, each dataset |
| P2 | C5 - B2 | alpha = 0.5 each dataset; alpha = 0.1 on CICIDS2017 and UNSW-NB15 |
| P3 | F - F0 and Fn - F0 | alpha = 0.1 and 0.5, CICIDS2017 and UNSW-NB15 |
| P4 | cost(F) - cost(B2) and cost(Fn) - cost(B2) | a = 0.1, CICIDS2017 and UNSW-NB15 |

## 7. Inference and wording

- Paired 95% t-interval of the mean difference over 10 seeds (9 df). A difference is **clear** when its interval excludes 0. The primary family is also reported with Holm-adjusted p-values (paired t-test) as a sensitivity analysis; unadjusted intervals remain the declared analysis.
- **No equivalence claims.** A difference that is not clear is reported as "no statistically clear difference", with its estimate and interval; never "tie", "adds nothing" or "the same". "Identical" is used only where the runs are bit-identical, which is checked (for example C1 and B2 when the detector fires at every boundary).
- **Practically relevant difference:** 0.02 macro-F1. Ten seeds per cell, as in protocol v1 and within the GPU budget. From the development spread of the paired differences (section 9), ten seeds give 80% power at the 5% level to detect 0.011-0.04 for most primary contrasts. Some reach only 0.08-0.31: C5-B2 on CICIDS2017 at both alpha levels and on UNSW-NB15 at 0.1, and F-F0/Fn-F0 on CICIDS2017 at 0.1. There a difference that is not clear is weak evidence of no effect, and the text must say so.
- Seed intervals describe these corpora and this randomisation, not other networks.

## 8. Conduct

- No code change after launch. A crashed run is resumed; the resume check refuses any settings change.
- R1 and R2 may run concurrently. The completion time and any deviation are appended below the launch record; nothing above this line changes after launch.
- Tables and figures are generated from the result files by `analyse_final_eval.py` and `make_final_tables.py`. The thesis prose is updated only after both runs finish.

## 9. Development evidence used to set this protocol (consumed seeds 42-44)

Development runs, repaired implementation, seeds 42-44 (n = 3), consumed
(`checkpoints/audit_repair_dev/`, summarised by `summarise_repair_dev.py`).
Paired SD is the standard deviation of the per-seed difference; "n=10" is the
difference detectable with 80% power at the 5% level (0.995 x SD).

| Contrast | Dev mean | Paired SD | Detectable at n = 10 |
|---|---:|---:|---:|
| CICIDS a=0.1 B2-B | +0.444 | 0.086 | 0.086 |
| CICIDS a=0.1 C5-B2 | -0.609 | 0.310 | 0.308 |
| CICIDS a=0.1 F-F0 | -0.298 | 0.267 | 0.266 |
| CICIDS a=0.1 Fn-F0 | -0.228 | 0.263 | 0.262 |
| CICIDS a=0.5 B2-B | +0.427 | 0.017 | 0.017 |
| CICIDS a=0.5 C5-B2 | -0.048 | 0.083 | 0.083 |
| CICIDS a=0.5 F-F0 | -0.162 | 0.079 | 0.079 |
| CICIDS a=0.5 Fn-F0 | -0.166 | 0.050 | 0.049 |
| UNSW a=0.1 B2-B | +0.232 | 0.027 | 0.027 |
| UNSW a=0.1 C5-B2 | -0.180 | 0.156 | 0.155 |
| UNSW a=0.1 F-F0 | -0.069 | 0.044 | 0.044 |
| UNSW a=0.1 Fn-F0 | -0.049 | 0.048 | 0.048 |
| UNSW a=0.5 B2-B | +0.220 | 0.038 | 0.038 |
| UNSW a=0.5 C5-B2 | +0.014 | 0.024 | 0.024 |
| UNSW a=0.5 F-F0 | -0.018 | 0.014 | 0.014 |
| UNSW a=0.5 Fn-F0 | -0.028 | 0.011 | 0.011 |
| 2018 a=0.5 B2-B | +0.650 | 0.024 | 0.024 |
| 2018 a=0.5 C5-B2 | -0.033 | 0.036 | 0.036 |
| 2018 a=0.5 F-F0 | -0.034 | 0.023 | 0.022 |
| 2018 a=0.5 Fn-F0 | -0.028 | 0.018 | 0.018 |

Stability checks on the same runs:
- No arm collapsed from the causal preprocessing on any dataset (CSE-CIC-IDS2018 rehearsal 0.81).
- The memory bound binds only on CSE-CIC-IDS2018 (peak 1,400 rows).
- C1 equals B2 exactly wherever the detector adapted at every boundary.
- The raw Eq. 8 target collapses the L_LS arms at alpha 0.1 on CICIDS2017, which is why both readings are run.

These numbers set only the practically relevant difference and the wording rule. They
are not reported as results.
