# Audit repair plan (started 2026-09-26)

Response to the independent audit `E:/thesis/audits/2026-09-26_thesis_draft/THESIS_AUDIT.md`.
The authors chose the **full-repair route, in stages**. Their constraints:

- Keep the introduction's wording and structure; make only necessary factual corrections.
- Preserve every original result file; report changes transparently (before/after).
- Development checks use already-consumed seeds. Then freeze the corrected implementation
  and protocol. Paired reruns on 47-56 give before/after diagnostics. Any new confirmatory
  claim selected through development is tested on fresh reserved seeds.
- Separate retrospective retention measurements from forward-stream performance.
- No new models, no scope expansion.
- Assistance disclosure (audit F11): at the authors' instruction, the thesis contains
  no such passage and makes no claim either way. Both earlier paragraphs are removed,
  and no attribution lines appear anywhere.

## Verified findings (code checked 2026-09-26)

| # | Finding | Status |
|---|---|---|
| F1 | Scaler fitted on all training concepts. CICIDS inf-fill uses each concept's full rows (incl. held-out rows) before the split. UNSW ordinal encoder and scaler fitted on the whole training file | confirmed; fix in all 3 loaders |
| F2 | "200 samples of the new attack" is wrong: `detect()` needs 200 confidence scores of any traffic and never sees labels | confirmed; withdraw the claim; fix `partition_properties.py` docstring |
| F3 | `train_ae` uses `fork_rng(devices=[])`, but `manual_seed` also reseeds CUDA and the CUDA state is not restored | confirmed; fix + regression checks |
| F4 | FL-IIDS Eq. 8 vs code: direction, target normalisation, teacher probabilities, first task, detachment, teacher choice | confirmed; see fidelity matrix below |
| F5 | Rehearsal caps per class per concept and never evicts. FL-IIDS keeps herding prefixes beyond the active quota | confirmed; enforce the bound, measure storage |
| F6-F10, F12 | Wording/claims | text edits |
| F11 | Disclosure of assistance | passages removed at the authors' instruction |

## FL-IIDS fidelity matrix (paper: Jin et al., FGCS 151, 2024)

| Element | Paper | v1 code | v2 decision |
|---|---|---|---|
| Memory size | fixed total m (2% of training set) | grows with the rehearsal arm's cap rule; stores full herding prefixes | m = min(L, rehearsal size); rows beyond the quota physically discarded |
| Allocation (Eq. 3) | proportional to class size over past tasks | same, largest-remainder rounding | unchanged |
| Exemplar choice | herding | iCaRL herding on L2-normalised penultimate features | unchanged |
| L_GB (Eqs. 4-7) | \|p_y - 1\|, averaged over new/old samples in the batch | same; weights detached | unchanged; detachment disclosed |
| L_LS (Eq. 8) | printed as KL(student ‖ smoothed label); smoothed = one-hot with the first K_old dims replaced by the old model's output | KL(normalised target ‖ student); teacher = full softmax sliced to old dims | A literal KL(student ‖ target) is infinite whenever the target has a zero entry, which happens for every sample with any class not yet seen. So the distillation direction (target ‖ student) is kept as the only computable reading and disclosed. Teacher = softmax over **old-class logits only**, the fixed-head analogue of the paper's K_old-output old model. Target normalisation is ambiguous: `raw` (sums to 1 or 2, as the text describes) vs `normalised` (v1). No author code is public, so both run as arms |
| First task | no old model; Eq. 9 undefined | 0.5 x L_GB, i.e. half the gradient scale of CE | disclosed; the loss-scale control below separates it |
| Old model without relay client | "each client stores the best-performing model of the current task's global rounds" | best of the task's R global states by accuracy on the client's own memory | unchanged; criterion disclosed |
| Participation, model, optimiser | 3 of 10 clients; CNN-GRU; SGD lr 1.0/0.1; 20 epochs | all 10; shared MLP; momentum 0.02; 10 epochs | unchanged (protocol alignment), disclosed |

Loss variants (explicit scaling, controlled RNG), run in development and then as frozen arms:
`ce` (1.0 CE, v1 F0), `ce_half` (0.5 CE), `gb` (0.5 GB), `ce_ls` (0.5 CE + 0.5 LS), `fliids` (0.5 GB + 0.5 LS).

## Repairs, with switches (legacy value reproduces the original results)

| Switch | Repaired | Legacy |
|---|---|---|
| `drift_channels.ISOLATE_CUDA_RNG` | True | False |
| `ids_loader/unsw_loader/ids2018_loader.PREPROC` | `'causal'`: non-finite fill, encoder and scaler fitted on the **initial concept's training rows only**, then frozen | `'all_train'` |
| `ids_cda_pipeline/unsw_cda_pipeline.MEMORY_BOUND` | True: after each concept, class-balanced random eviction to at most L rows (per-client seeded RNG) | False |
| `fl_iids.MEMORY_BOUND`, `LS_TARGET`, `TEACHER` | True, chosen in dev, `'old_only'` | False, `'normalised'`, `'full'` |

`final_eval.py --impl legacy|repaired` sets all of these together.

New measurements in `--impl repaired` runs:
- forward-stream performance: each arriving concept's held-out rows, evaluated just before the model trains on that concept
- physical memory: rows and bytes per client, including model copies
- reproducibility metadata: source-file hashes, package versions, dataset fingerprints, effective detector/memory/preprocessing settings, command line. Resume refuses a file written with different settings.

## Seeds

Used: 0-10, 41-56, 100-104, 200-209. The audit reran 47-49.
Development: consumed seeds (41-46).
Paired before/after: 47-56.
**Reserved for confirmation: 600-609** (never used).

## Stages

1. [x] Code repairs + unit checks (`test_repairs.py`; existing suites still pass)
2. [x] Equivalence check: C with every boundary triggered == B2, bit for bit (repaired);
       legacy RNG differs by 0.023 on the small check. Legacy mode reproduces the saved
       rehearsal scores exactly (CICIDS and UNSW, seed 47).
   Decisions taken by argument, before any score: TEACHER = 'old_only';
   CSE-CIC-IDS2018 follows `ids_loader.PREPROC`. The Eq. 8 target: 'raw' was
   declared first, as the paper's description. The first dev seed then showed that
   arms using L_LS collapse under it (CICIDS a=0.1: 0.50 and 0.38). A KL divergence
   is defined only between distributions, which argues for 'normalised', and no
   author code is public. So rather than choose by score, BOTH readings run as
   arms: `fliids`/`fliids_ce_ls` (raw) and `fliids_norm`/`fliids_ce_ls_norm`
   (normalised). Claims about FL-IIDS must hold under both or be stated per reading.
   Development queue: `run_dev_queue.sh` -> `checkpoints/audit_repair_dev/`.
3. [x] Development runs on consumed seeds (`summarise_repair_dev.py` prints them). What
       each repair moved (seeds 42-44; 42-43 for 2018; mean change from legacy):
       - causal preprocessing: CICIDS B2 -0.071, C -0.022, F -0.137; UNSW about -0.02;
         2018 +0.008 to +0.024. The only repair that matters materially.
       - memory bound: 0 on CICIDS/UNSW (memory never exceeded L); 2018 B2 +0.011
       - CUDA RNG isolation (C only): -0.0006 CICIDS, +0.0015 UNSW, -0.007 2018
       - FL-IIDS teacher/target: +0.001 CICIDS, -0.003 UNSW, +0.004 2018
4. [x] Freeze: FINAL_EVAL_PROTOCOL_v2.md, sha e12a2782..., launch record
       `checkpoints/final_eval_v2_launch.json` (2026-09-26)
5. [x] Completed 2026-09-27 17:30. Two GPU faults (another workload started on the machine)
       were resumed; the crashed cell was re-run independently and was bit-identical
       (12/12 arms). The 2018 part was split by seed at the user's request (scheduling
       only). All deviations are logged in the launch record.
6. [x] Analysis: `final_v2_R2_analysis.json` (claims) and `final_v2_R1_analysis.json`
       (before/after). Holm family corrected to include P4 (logged deviation). Tables and
       figures regenerated (`make_final_tables.py`, `make_final_figs.py`).
7. [x] Thesis text updated (Ch1 methodology, Ch3 risk rows and economics, Ch4, Ch5, Ch6,
       abstract, App. B). A verification workflow confirmed 15 findings; 14 were fixed
       and the last is the co-supervisor placeholder (needs the author). thesis_latex
       commit 3387f28; PDF and Overleaf zip refreshed 2026-09-27.
   Still open: PR the repaired code and results to CDA-FedAvg-IDS (only when the user
   asks); the co-supervisor designation; the approval semester wording.
(earlier note) Paired rerun 47-56 (R1) and confirmation run 600-609 (R2), running:
       `run_final_v2.sh R1|R2` -> `checkpoints/final_v2/`. The user chose to leave
       the scheduling as it is (no extra parallel jobs).
6. [ ] Analysis, before/after accounting, tables, figures
7. [ ] Thesis text: claim corrections (F1-F12), results, objectives.
   Done so far: Ch1 targeted fixes (RQs/objectives unchanged), Ch2 literature (FedDrift,
   CDA-FedAvg protocol, FedAvg non-IID, Winter, Table 2.2 criteria, gap wording), Ch3
   (requirements, energy, standards), ethics statement, Appendix A.
   Pending after the reruns: Ch1 methodology paragraph, Table 3.1 new risk rows,
   Ch4 methods, Ch5 results, Ch6, abstract, Appendix B (F12).
   **User: make no further chapter changes until the runs finish.** The Ch4 edits
   were drafted and reverted; they are saved in
   `checkpoints/audit_repair_dev/pending_ch4_edits.py`, to apply after the runs.
   Run plan chosen by the user: FULL (complete breakdown; paired rerun on 47-56 with
   every arm and all alphas; confirmation on 600-609 with all alphas).
