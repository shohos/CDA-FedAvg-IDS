# Where each thesis result comes from

Numbering follows the final thesis report. "Analysis" means a file written by step 2 of
the README from the result files beside it. All result files are under `checkpoints/`.

Arm keys used in the result files:

| Key | Arm in the thesis |
|---|---|
| `iid_pooled` | Pooled reference |
| `iid_stream` | Shuffled stream |
| `nonstat` | No memory |
| `rehearsal` | Always-adapt rehearsal |
| `cda` | CDA-FedAvg, quorum of five |
| `cda_q1` | CDA-FedAvg, any-one rule |
| `fliids`, `fliids_norm` | Adapted FL-IIDS, raw and normalised label-smoothing target |
| `fliids_ce` | Adapted FL-IIDS, plain loss |
| `fliids_ce_half`, `fliids_gb`, `fliids_ce_ls`, `fliids_ce_ls_norm` | the loss ablation of Section 5.2.6 |

## Chapter 4: method

| Thesis item | Script | Result file |
|---|---|---|
| Table 4.2 (datasets), Table 4.3 (what each concept adds) | `ids_loader.py`, `ids2018_loader.py`, `unsw_loader.py` | the loaders print the row counts; each run's `concepts` entry records the classes per concept |
| Table 4.4 (measured properties of the client splits) | `partition_properties.py --draws 200` | `partition_properties.json` |
| Figure 4.5 (a real CICIDS2017 split) | `make_thesis_figures.py` | `figures/thesis/fig_dirichlet_split.png` |
| Figure 4.6, parameter counts (62,727 and 53,706) | the model in `ids_cda_pipeline.py` / `unsw_cda_pipeline.py` | |
| Table 4.5 (training settings) | module constants of the pipelines, `drift_channels.py`, `memory_bound.py` | `meta.config` of every final-evaluation file |
| Section 4.6, detector and memory | `drift_channels.py`, `memory_bound.py`, `detector_faithfulness.py` (the confidence channel against Algorithm 5 of the source paper; checked in `test_pipeline.py`) | |
| Section 4.7, adapted FL-IIDS | `fl_iids.py` | |
| Section 4.9, other combining rules | `aggregators.py` (current, SCAFFOLD repaired); `legacy/aggregators_pre_repair.py` (the version behind the SCAFFOLD (orig.) results) | |
| Section 4.10, TabFM front end | `tabfm_frontend.py`, `tabfm_main_eval.py` | |
| Section 4.11, the four code corrections | `final_eval.py --impl legacy` / `--impl repaired`; development in `dev_repairs.py` | `audit_repair_dev/`, summarised by `summarise_repair_dev.py` in `audit_repair_dev/summary.json` |
| Figures 4.1–4.4, 4.7–4.11 | drawn diagrams, not generated from results | |

The frozen plans are in `docs/protocols/`: `FINAL_EVAL_PROTOCOL.md` (v1, seeds 47–56),
`FINAL_EVAL_PROTOCOL_v2.md` (corrected code, R1 and R2), `FINAL_EVAL_PROTOCOL_v3.md`
(combining rules), `AUDIT_REPAIR_PLAN.md` (the corrections and their checks) and
`AGG_STUDY_PLAN.md`; the TabFM protocol is
`checkpoints/tabfm_integration/TABFM_INTEGRATION_PROTOCOL.md`. Launch records with each
plan's SHA-256: `final_eval_launch.json`, `final_eval_v2_launch.json`,
`agg_v3_launch.json`.

## Chapter 5, Section 5.1: development (exploratory, seeds 42–46)

These runs used earlier versions of the shared code (see the README). Each file stores its
settings in `meta`; where a record stored only the typed command, its settings were
recovered from the run logs.

| Thesis item | Script | Result file |
|---|---|---|
| Table 5.1 (training steps per arm, original CICIDS2017 release) | `audit_exposure.py` | printed |
| "about 37 times" on CSE-CIC-IDS2018 | `run_pooled_arm.py` | `final_v2/R2_ids2018_pooled.json` (366,000 steps) against `nonstat` steps in `final_v2_R2_analysis.json` |
| Section 5.1.2, alarms on the stream without drift; clients alarming per concept | `null_control.py` | `null_heldout.json` (α = 0.5), `null_a01.json` (α = 0.1, including the quorum-of-five and size-weighted rules), `null_improved.json` (corrected labels), `null_control.json` |
| Section 5.1.2, size-weighted alarm rule | `alpha_sweep_v2.py` | `alpha_qweight.json`, `null_a01.json` |
| Section 5.1.2, Thursday-afternoon session (eight-session layout) | `baseline_v2.py --dataset cicids --concept-mode 8session` | `cicids_8session_q5.json` |
| Section 5.1.2, 36 of 40 and 44 of 50 changes on CSE-CIC-IDS2018 | `final_eval.py` (R2) | detection records of the `cda` arm in `final_v2/R2_ids2018.json` |
| Section 5.1.3, memory size | `memory_sweep.py` | `memory_sweep.json`, `memory_sweep_v2.json` |
| Section 5.1.3, momentum | `opt_sweep.py` | `opt_sweep.json` |
| Section 5.1.3, gradient clipping | `constant_sweep.py` | `constant_sweep.json` |
| Section 5.1.3, layer against batch normalisation at each one's best learning rate | `lr_sweep.py` | `lr_sweep.json` |
| Section 5.1.3, corrected CICIDS2017 release | `baseline_v2.py` on both releases | `best_config_v2.json` (original labels), `cicids_improved.json` (corrected) |
| Development runs across α and on CSE-CIC-IDS2018 | `alpha_sweep_v2.py`, `run_2018.py --concepts daily` | `alpha_sweep_v2.json`, `alpha_sweep_final.json`, `alpha_improved.json`, `alpha_quorum5.json`, `ids2018_daily_v2.json` |
| Section 5.6, published configuration 0.483 to final 0.974 | `baseline_v2.py` | `baseline_v2_alpha05.json`, `baseline_v2_layernorm.json`, `best_config_v2.json`, `cda_best.json`, `cicids_improved.json` |
| FL-IIDS learning rate 0.02 (seed 41) | `final_eval.py` | `dev41_*.json`; rule and means in `final_eval_launch.json` |

## Chapter 5, Section 5.2: final evaluation (corrected code)

Unless stated, from the confirmation run R2 (seeds 600–609): result files
`final_v2/R2_cicids.json`, `R2_unsw.json`, `R2_ids2018.json`; analysis
`final_v2_R2_analysis.json`. Contrasts are paired by seed with 95% t-intervals (9 degrees
of freedom); contrasts in the plan's primary family also carry a Holm-adjusted p-value
(`p_holm_primary`).

| Thesis item | Analysis field / file |
|---|---|
| Table 5.2 (corrected minus original code, seeds 47–56) | `final_v2_R1_analysis.json` (R1 against `final_eval_analysis.json`, which is computed from `final_*.json`) |
| Table 5.3 (scores of the arms) | `scores`; pooled reference: CICIDS2017 and UNSW-NB15 from `cicids_improved.json` and `unsw_final_q5.json` (development, seeds 42–46), CSE-CIC-IDS2018 from `final_v2/R2_ids2018_pooled.json` |
| Table 5.4 (key differences) | `contrasts` (B2-B, C1-B2, C5-B2, B2-A2) |
| Table 5.5 (training steps, largest memory) | `cost`, `memory`; pooled steps from `final_v2/R2_ids2018_pooled.json` |
| Forward-stream score, one in nine attack flows flagged | `forward` (`macro_f1`, `detection`) |
| Table 5.6 (cost of heterogeneity) | `noniid_cost` |
| Table 5.7 (detection against always adapting) | `scores`, `contrasts` (C5-B2, C1-B2), `adaptations`, `cost` |
| Table 5.8 (class-level forgetting) | `forgetting` |
| Table 5.9 (adapted FL-IIDS) | `contrasts` (F-B2, Fn-B2, F-F0, Fn-F0) |
| Loss ablation in Section 5.2.6 | `contrasts` (Fgb-F0h, Fls-F0h, Flsn-F0h) |
| Table 5.10 (combining rules, seeds 800–809) | `agg_analysis.json` from `agg/<rule>_<dataset>.json` (`analyse_agg.py`); FedProx weight from `agg_dev/`; the crashed and rerun cell in `agg_crashcheck/` (`compare_crash_cell.py`) |
| Table 5.11, Figure 5.4 (TabFM front end, seeds 42–46) | `tabfm_integration/tabfm_summary.json`, `tabfm_per_seed.csv` (`analyse_tabfm_integration.py`) from the three `tabfm_dev_*.json` files |
| TabFM as a classifier without training (0.928 / 0.989 / 0.991 and 0.398 / 0.454 / 0.439) | `tabfm_probe.json` (`tabfm_probe.py`) |
| Table 5.12 (final comparison) | assembled from `final_v2_R2_analysis.json`, `tabfm_summary.json` and `tabfm_probe.json` |
| Section 5.5 (statistical analysis) | `p_holm_primary` in `contrasts` of `final_v2_R2_analysis.json` and `agg_analysis.json` |
| Figures 5.1 and 5.2 | `make_thesis_figures.py` → `figures/thesis/fig_res_fivearm.png`, `fig_res_heterogeneity.png` |
| Figure 5.3 (recall of each class after every round) | a separate run on seeds 47–56 with the original code; its per-round records are not in this repository |

A plain-text digest of the R2 analysis, cell by cell, is in `final_v2/FACTS_R2.txt`.
