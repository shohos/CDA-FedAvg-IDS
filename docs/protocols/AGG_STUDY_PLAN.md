# Aggregator study (started 2026-09-27)

Question, from the supervisor via the authors: which server aggregation rule works
best for the non-IID CDA scenario? Candidates are FedAvg (already done), FedProx,
SCAFFOLD and coordinate-wise Median, at low and high heterogeneity.

Scope chosen by the authors:
- arms: rehearsal B2 (the CDA pipeline, always adapt), detector C with quorum 5,
  and no memory B;
- datasets: CICIDS2017, UNSW-NB15 and CSE-CIC-IDS2018;
- alpha in {0.1, 0.5, 100};
- tune on consumed seeds, then fresh seeds; results go into the thesis.

Implementation: `aggregators.py`, wired into both pipelines;
`final_eval.py --impl repaired --agg {fedavg,fedprox,scaffold,median} [--mu]`.
FedAvg is untouched: it reproduces the confirmation run bit for bit (checked on
CICIDS alpha 0.5 seed 600, arms B, B2, C5).

Implementation choices, fixed before any tuning:
- **FedProx:** proximal term (mu/2)||w - w_global||^2 in the local loss; weighted
  average at the server.
- **SCAFFOLD:** gradient correction (c - c_i). c_i uses option I (local gradient at
  the server model, independent of our momentum optimiser); c <- c + (1/N) sum dc_i;
  server lr 1; sample-weighted model average so that only the control variates differ
  from FedAvg.
- **Median:** coordinate-wise, unweighted; mean of the two middle values when the
  count is even.
- The rest is the repaired implementation (protocol v2), FL-IIDS lr irrelevant.

## Development (consumed seeds 42-44)

Rehearsal arm B2 only, CICIDS2017 and UNSW-NB15, alpha in {0.1, 0.5, 100}:
- FedProx with mu in {0.001, 0.01, 0.1, 1.0};
- FedAvg, SCAFFOLD and Median once each, as a sanity check.

**mu selection rule, fixed before the runs:** the mu with the highest mean B2
macro-F1 over the 18 cells (2 datasets x 3 alphas x 3 seeds); within 0.002 of the
best, take the smaller mu. SCAFFOLD and Median have no tuned setting. CSE-CIC-IDS2018
is not used for tuning (it is slow), so its results test the transfer of the choice.

## Confirmation

Fresh seeds **800-809**, checked unused on 2026-09-27: no result file uses them, and
no script uses them as a seed. The run follows FINAL_EVAL_PROTOCOL_v3.md, written
after development and before any confirmation run.
