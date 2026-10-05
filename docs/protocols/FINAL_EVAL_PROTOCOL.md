# Final evaluation protocol (thesis Chapter 5)

Written 25 September 2026, **before** any evaluation seed below is run. The
development runs on seed 41 are used only to pick FL-IIDS's learning rate (rule
below). Nothing in this file is changed after the evaluation runs start; the
file's SHA-256 is recorded at launch.

## Purpose

1. **Validate** the headline results of Chapter 5 on fresh seeds that played no
   part in developing the configuration (all earlier corpus results used seeds
   42–46, which were also used to choose settings).
2. **Complete objective O5**: compare against FL-IIDS (Jin et al., FGCS 151,
   2024) on identical streams, across the heterogeneity range, to measure how
   much its IID assumption costs it under non-IID clients.
3. **Complete objective O6's forgetting half**: track every class from the
   concept it first appears in to the end of the stream.

## Frozen configuration (every arm)

LayerNorm DNN; SGD with momentum 0.9, learning rate 0.01; batch 100; 10 local
epochs; clip norm 10; 10 clients, full participation; 5 rounds per concept;
memory L = 1400 with the per-class cap rule; corrected CICIDS2017 labels;
5,000 records per class per concept. Code as of this date: `final_eval.py`,
`fl_iids.py`, and the unchanged pipeline modules.

## Arms

| id | arm |
|---|---|
| A2 | `iid_stream`: compute-matched IID ceiling |
| B | `nonstat`: ordered stream, no memory |
| B2 | `rehearsal`: memory, adapts at every concept boundary |
| C5 | `cda`: memory + hybrid detector, quorum 5 (the final configuration) |
| C1 | `cda` with quorum 1 (the source method's any-one rule) |
| F | `fliids`: FL-IIDS, see `fl_iids.py` |
| F0 | `fliids_ce`: FL-IIDS's memory and schedule with plain cross-entropy |

FL-IIDS's learning rate: the value in {0.01, 0.02, 0.05} with the highest
seed-41 macro-F1 averaged over CICIDS2017 and UNSW-NB15 at α = 0.5; ties go to
0.01. F0 uses the same learning rate as F, so F − F0 isolates the loss. All
other arms keep 0.01.

## Seeds, datasets, heterogeneity

- **Evaluation seeds 47–56** (ten). Checked unused on 25 September 2026: the
  checkpoint store records seeds 0–9, 41–46 only.
- CICIDS2017 (5 days) and UNSW-NB15 (5 concepts): α ∈ {0.1, 0.5, 1.0, 100}.
- CSE-CIC-IDS2018 (10 days): α = 0.5 only. The loader's own seed is now set per
  seed (earlier 2018 runs left it at 42, so their three seeds shared one
  partition and test split).

## Declared quantities

Primary endpoint: final macro-F1 on the held-out test set. For each quantity:
mean over the ten seeds, the sample standard deviation, and for paired
differences a 95% t-interval (9 degrees of freedom). A difference is called
"clear" only when its interval excludes zero. No correction for multiple
comparisons is applied; this is stated with the results, and every declared
quantity is reported whatever its direction.

1. Validation at α = 0.5, each corpus: B2 − B (value of rehearsal), C5 − B2 and
   C1 − B2 (value of detection), B2 − A2 (rehearsal against the matched IID
   ceiling).
2. O5, each α on CICIDS2017 and UNSW-NB15 (and α = 0.5 on 2018): F − B2 and
   F − C5; and the cost of non-IID for each method, score(α = 100) −
   score(α), for F and for B2.
3. What FL-IIDS's losses add: F − F0 at each α.
4. Forgetting (arms B, B2, C5, C1, F, F0): for each class, its recall right
   after the concept that introduces it, its best recall afterwards, and its
   final recall. Forgetting of a class = best recall at any concept end from its
   introduction up to (not including) the last concept, minus final recall,
   floored at 0. Reported as the mean over classes introduced before the last
   concept, plus the number of classes that were learned (recall ≥ 0.5 at some
   point) and later lost (final recall < 0.5). The same with per-class F1 is
   reported alongside.
5. Cost: gradient steps per arm, and memory records per client.

## What would count against our earlier claims

- C5 − B2 clearly positive at α = 0.5 on any corpus would contradict "detection
  adds nothing over always adapting".
- B2 − B not clearly positive would contradict "rehearsal carries the benefit".
- These are reported as found.
