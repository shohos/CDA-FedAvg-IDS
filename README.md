# Concept Drift-Aware Federated Learning for Network Intrusion Detection under Non-IID Data Distributions

This repository holds the code, result files and frozen evaluation protocols behind the
B.Sc. thesis of the same name (Department of Computer Science and Engineering, Brac
University, 2026), by Razeen Hassan, Tawfiqur Rahman, Khondkar Nahiyan Karim and Shoaib
Hossain, supervised by Dr. Jannatun Noor Mukta and co-supervised by Dr. Md. Sakir Hossain.

Every number in the thesis's final evaluation can be regenerated from the files here, and
every experiment can be rerun from the scripts here once the public datasets are in place.
[`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md) lists, table by table and figure by figure,
which script and which result file stand behind each reported result.

## The study in brief

The study adapts CDA-FedAvg, a federated learning method that detects concept drift and
retrains on a memory of past examples, to flow-based network intrusion detection. It
evaluates it on CICIDS2017 (corrected labels), CSE-CIC-IDS2018 and UNSW-NB15 with ten
clients whose traffic ranges from near-identical (Dirichlet α = 100) to very different
(α = 0.1). A five-arm design separates the memory from the drift detector:

| Arm | What it does |
|---|---|
| Pooled reference | every client retrains on all concepts mixed together; sees no drift, trains 19–37 times as much |
| Shuffled stream | the same mixed data, one slice per round; trains about as much as the arm without memory |
| No memory | concepts arrive in time order; trains on the current concept only |
| Always-adapt rehearsal | adds each new concept to a bounded memory (L = 1,400 rows) and retrains at every change |
| CDA-FedAvg | the same memory, retrains only when the server declares drift (any-one rule or quorum of five) |

An adapted FL-IIDS, three other ways of combining client models and a frozen TabFM
front end complete the study.

**Main results** (confirmation run on seeds 600–609, α = 0.5, macro-F1; "clear" means the
paired 95% confidence interval over ten seeds excludes zero):

| Comparison | CICIDS2017 | CSE-CIC-IDS2018 | UNSW-NB15 |
|---|---|---|---|
| Always-adapt rehearsal − no memory | +0.436 (clear) | +0.645 (clear) | +0.217 (clear) |
| CDA-FedAvg, any-one rule − always-adapt | identical | identical | identical |
| CDA-FedAvg, quorum of five − always-adapt | −0.028 (not clear) | −0.020 (not clear) | identical |

- With very different clients (α = 0.1), the quorum-of-five rule met its quorum at only 14
  of 40 changes on CICIDS2017 and lost 0.380 against always adapting, a clear loss that
  also holds after the Holm correction.
- From near-identical to very different clients, the adapted FL-IIDS lost 0.44–0.49 on
  CICIDS2017 and 0.15–0.19 on UNSW-NB15, against 0.06 and no clear loss for rehearsal.
- FedProx showed no demonstrated benefit over FedAvg (seeds 800–809). The coordinate-wise
  median was worse with very different clients on all three datasets by the unadjusted
  intervals, and on CICIDS2017 only after the Holm correction. The SCAFFOLD columns are
  diagnostics of the original implementation, not evidence about SCAFFOLD.
- The frozen TabFM front end raised macro-F1 by 0.066 (CICIDS2017) and 0.068 (UNSW-NB15)
  with always-adapt rehearsal on five development seeds. This is exploratory; the
  classifier still trains, so adaptation without training remains open.

## Repository layout

```
*.py, *.sh          all code, deliberately flat: modules import each other from this
                    directory and read and write checkpoints/ relative to it
checkpoints/        every result file (JSON) behind the thesis, plus launch records
  final_*.json        first final evaluation, original code, seeds 47-56
  final_v2/           corrected code: R1 (seeds 47-56) and R2 confirmation (600-609)
  final_v2_R*_analysis.json, final_eval_analysis.json   the computed statistics
  audit_repair_dev/   development of the code corrections (seeds 41-46)
  agg/, agg_dev/      other ways of combining client models (800-809; dev 42-44)
  tabfm_integration/  TabFM front-end runs, summary and protocol (seeds 42-46)
  *.json              development runs behind thesis Section 5.1
docs/RESULTS_MAP.md every thesis table and figure -> script -> result file
docs/protocols/     the evaluation plans, each frozen before its runs started
figures/thesis/     the result charts of the thesis, from make_thesis_figures.py
legacy/             code archived because a later repair changed it (see below)
```

## Setup

Python 3.11 and an NVIDIA GPU. The thesis runs used Python 3.11.9, PyTorch 2.7.1 with
CUDA 11.8 and one RTX 3060 Laptop GPU; every result file records the exact versions.

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

TabFM (`tabfm`, only needed for the TabFM extension) downloads about 3.3 GB of weights
on first use.

## Data

The datasets are not redistributed. Place them next to the code under these names:

| Directory | Dataset | Source |
|---|---|---|
| `CICIDS2017-improved/` | CICIDS2017, corrected labels (Engelen et al., 2021): the five `*-WorkingHours.csv` day files | <https://intrusion-detection.distrinet-research.be/WTMC2021/> |
| `MachineLearningCVE/` | CICIDS2017, original labels (`*.pcap_ISCX.csv`). Required: the loader reads its column names to map the corrected release onto them; the files are also used for the development comparison of the two releases | <https://www.unb.ca/cic/datasets/ids-2017.html> |
| `CICIDS2018/` | CSE-CIC-IDS2018, the ten `*_TrafficForML_CICFlowMeter.csv` day files (about 6.5 GB) | `bash download_ids2018.sh` |
| `CSV Files/Training and Testing Sets/` | UNSW-NB15, `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` | <https://research.unsw.edu.au/projects/unsw-nb15-dataset> |

Every final-evaluation result file records the SHA-256 of each data file it read
(`meta.config.data_sha256`), so you can confirm you have the same files.

## Reproducing the thesis

### 1. Check the code and the record

```bash
python test_eval_metrics.py && python test_pipeline.py && python test_repairs.py
python test_aggregators.py && python test_scaffold_repair.py
python test_repairs.py --data     # adds checks on CICIDS2017 and UNSW-NB15 (needs the datasets)
python verify_fingerprints.py
```

The five suites need no dataset. `verify_fingerprints.py` compares the code hashes stored
in the result files and launch records with the files in this directory. The repository
stores every file byte for byte (`.gitattributes`), so the hashes can be checked from a
fresh clone. Expect these differences:

- **R1 and R2** (seeds 47–56 and 600–609) were run before the aggregation study added
  hooks to `final_eval.py` and both pipelines, so those three files show as changed. With
  FedAvg the hooks are inactive: the current code reproduces two arms (always-adapt
  rehearsal and CDA-FedAvg with the any-one rule) of an R2 CICIDS2017 cell (seed 600,
  α = 0.5) bit for bit, histories and step counts included
  (`checkpoints/final_v2/verify/current_code_reproduction.json`). The v2 launch record
  also lists `analyse_final_eval.py`, which later gained fields for the aggregation study;
  it still reproduces the R1 and R2 analyses byte for byte (step 2).
- **The aggregation study** ran before SCAFFOLD was repaired. Its exact code is archived in
  `legacy/aggregation_study/`; `python verify_fingerprints.py --code legacy/aggregation_study checkpoints/agg/fedavg_cicids.json`
  reports every file as matching.

### 2. Regenerate every analysis from the saved results (no GPU, a few minutes)

```bash
# statistics: paired 95% intervals, Holm correction, forgetting, cost, forward-stream scores
python analyse_final_eval.py final_cicids.json final_unsw.json final_ids2018.json
python analyse_final_eval.py --out final_v2_R1_analysis.json --before final_eval_analysis.json \
    final_v2/R1_cicids.json final_v2/R1_unsw.json final_v2/R1_ids2018.json
python analyse_final_eval.py --out final_v2_R2_analysis.json \
    final_v2/R2_cicids.json final_v2/R2_unsw.json final_v2/R2_ids2018.json
python analyse_agg.py                     # -> checkpoints/agg_analysis.json
python analyse_tabfm_integration.py       # -> tabfm_integration/tabfm_summary.json, tabfm_per_seed.csv
python summarise_repair_dev.py            # -> audit_repair_dev/summary.json

# tables and figures
python make_final_tables.py tables        # LaTeX tables of the final evaluation
mkdir -p out/tables out/images && python make_agg_outputs.py out   # aggregation tables and figure
python make_thesis_figures.py             # -> figures/thesis/ (the split figure needs CICIDS2017)
```

Running these on the committed result files reproduces the committed analysis files and
the three result charts byte for byte.

### 3. Rerun the experiments (GPU)

Times are for the RTX 3060 Laptop GPU used in the thesis.

| What | Command | Time |
|---|---|---|
| First final evaluation, original code, seeds 47–56 (protocol v1) | `python final_eval.py --impl legacy --dataset cicids --alphas 0.1,0.5,1.0,100 --seeds 47,48,49,50,51,52,53,54,55,56 --arms iid_stream,nonstat,rehearsal,cda,cda_q1,fliids,fliids_ce --fliids-lr 0.02 --out final_cicids.json`, then the same for `unsw` and for `ids2018` with `--alphas 0.5` | 11 h |
| Corrected code: paired rerun R1 (47–56) and confirmation R2 (600–609) (protocol v2) | `bash run_final_v2.sh R1`, `bash run_final_v2.sh R2`, then `python merge_v2_parts.py R1 R2` (CSE-CIC-IDS2018 ran in two parts) | 22 h each |
| Pooled reference on CSE-CIC-IDS2018, R2 seeds | `python run_pooled_arm.py --dataset ids2018 --alphas 0.5 --seeds 600,601,602,603,604,605,606,607,608,609 --out final_v2/R2_ids2018_pooled.json` | 2 h |
| Development of the code corrections (seeds 41–46) | `bash run_dev_queue.sh`, `bash run_dev_queue2.sh` | 3 h |
| Other combining rules (protocol v3): development, then seeds 800–809 | `bash run_agg_dev.sh cicids`, `bash run_agg_dev.sh unsw`, then `bash run_agg_final.sh <rule> cicids unsw ids2018` for `fedavg`, `fedprox`, `median` | 2 h, then 28 h |
| SCAFFOLD (orig.) diagnostics | in a separate copy of the repository, copy `legacy/aggregation_study/*.py` over the top-level files, then `bash run_agg_final.sh scaffold cicids unsw ids2018` | (in the 28 h) |
| TabFM front end (seeds 42–46, α = 0.5) | `python tabfm_main_eval.py --dataset cicids --seeds 42,43,44,45,46 --alpha 0.5 --out checkpoints/tabfm_integration/tabfm_dev_cicids_42_46_a05.json`, and the same for `unsw` | a few hours |
| TabFM as a classifier without training (one pooled run) | `python tabfm_probe.py --dataset both --test-n 6000` | under 1 h |
| Development results of Section 5.1 | see [`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md) | varies |

Output names are relative to `checkpoints/` (for `tabfm_main_eval.py`, give the path from the repository root, as above).
`final_eval.py` resumes an interrupted file and refuses to resume one written with
different code or settings. The FL-IIDS learning rate (0.02) was chosen on seed 41
(`checkpoints/dev41_*.json`, rule in `final_eval_launch.json`).

## Seeds

Every seed set used for a confirmatory claim was checked unused before its runs started,
and each plan in `docs/protocols/` was fixed before its seeds were run.

| Seeds | Role |
|---|---|
| 42–46 | development and settings; exploratory (CSE-CIC-IDS2018 and the sweeps used three of them) |
| 41 | FL-IIDS learning rate; development of the code corrections (with 42–46) |
| 47–56 | first final evaluation (original code) and its paired rerun R1 (corrected code) |
| 600–609 | confirmation run R2: the main conclusions of the thesis |
| 800–809 | other ways of combining client models (FedProx weight chosen on 42–44) |

## Limits of what this repository reproduces

- **Development results** (thesis Section 5.1) come from earlier code. Shared modules have
  since been corrected, so rerunning a development script may not give the saved numbers
  exactly; the saved files are the record, and the thesis treats them as exploratory.
- **Figure 5.3** (recall of each class after every round) comes from a separate run on
  seeds 47–56 with the original code. The result files here keep one record per concept,
  not per round, so that figure cannot be redrawn from them.
- **Appendix A** (sharing attack knowledge without labels) is a separate experiment that no
  main result depends on; its code is not part of this repository.
- The federation is simulated on one machine with every client in every round.

## Citation

```bibtex
@thesis{hassan2026cdafedavg,
  title  = {Concept Drift-Aware Federated Learning for Network Intrusion Detection under Non-IID Data Distributions},
  author = {Hassan, Razeen and Rahman, Tawfiqur and Karim, Khondkar Nahiyan and Hossain, Shoaib},
  school = {Department of Computer Science and Engineering, Brac University},
  type   = {B.Sc. thesis},
  year   = {2026}
}
```
