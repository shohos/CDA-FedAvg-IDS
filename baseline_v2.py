"""
baseline_v2.py — corrected, compute-matched, multi-seed baseline at α=0.5.

Why this exists
---------------
audit_exposure.py shows the three published configurations do not share a
training budget: IID performs ~19x and CDA ~2.1x the gradient steps of the
non-stationary baseline. So "IID − non-stationary" measures forgetting *plus* a
large optimisation advantage, and "CDA − non-stationary" measures rehearsal
*plus* extra optimisation. Neither difference is attributable to one mechanism.

Five configurations, so every claim isolates one variable:

  A   IID-pooled     retrains on the whole pooled shard each round (as published;
                     ~19x budget). Kept for continuity — it is a centralised-
                     equivalent ceiling, not a fair federated baseline.
  A2  IID-stream     same 25-slice schedule as B, ordering shuffled away.
                     Step-for-step matched to B: A2 − B is forgetting, alone.
  B   Non-stationary concepts streamed in order, no memory.
  B2  Rehearsal      the CDA buffer with an oracle detector (adapt at every
                     concept boundary). B2 − B is what the buffer is worth.
  C   CDA-FedAvg     buffer + Beta-CUSUM detector. C − B2 is what the detector
                     is worth, and it can be negative when drifts are missed.

RNG is reseeded before every configuration, so no configuration inherits the
RNG state left by the one before it (that alone flipped a drift detection in
the α-sweep). Data is loaded once per (seed, dataset), so all five see the
identical partition.

Output: checkpoints/baseline_v2_alpha05.json
"""

import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ids_loader, unsw_loader
from ids_loader import load_ids_dataset, CLASS_NAMES, CKPT_DIR
from ids_cda_pipeline import (run_iid_fedavg, run_iid_stream_fedavg,
                              run_nonstationary_fedavg, run_cda_fedavg_ids,
                              N_CLIENTS)
from unsw_loader import load_unsw_dataset, CLASS_NAMES_UNSW
from unsw_cda_pipeline import (run_iid_fedavg_unsw, run_iid_stream_fedavg_unsw,
                               run_nonstationary_fedavg_unsw, run_cda_fedavg_unsw)

ALPHA = 0.5
SEEDS = [42, 43, 44, 45, 46]
CONFIGS = ['iid_pooled', 'iid_stream', 'nonstat', 'rehearsal', 'cda']
DETECTOR = 'hybrid'  # best measured: the confidence channel fires 0/4 under
                     # LayerNorm (a stable model has stable confidence), so
                     # 'conf' is not a usable default any more. 'hybrid' adds the
                     # reconstruction channel; see drift_channels.py.
FUSION = 'or'        # 'or' | 'and'


def _row(res):
    return {k: (None if isinstance(v, float) and v != v else v)
            for k, v in res[-1].items()}


def run_one(which, concepts, test_set, device, seed, configs=None):
    """Run the requested configurations on one loaded dataset."""
    configs = configs or CONFIGS
    if which == 'cicids':
        runners = {
            'iid_pooled': lambda: run_iid_fedavg(concepts, test_set, device, verbose=False),
            'iid_stream': lambda: run_iid_stream_fedavg(concepts, test_set, device, verbose=False),
            'nonstat':    lambda: run_nonstationary_fedavg(concepts, test_set, device, verbose=False),
            'rehearsal':  lambda: run_cda_fedavg_ids(concepts, test_set, device, verbose=False, always_adapt=True),
            'cda':        lambda: run_cda_fedavg_ids(concepts, test_set, device, verbose=False,
                                                       detector_mode=DETECTOR, fusion=FUSION),
        }
    else:
        runners = {
            'iid_pooled': lambda: run_iid_fedavg_unsw(concepts, test_set, device, verbose=False),
            'iid_stream': lambda: run_iid_stream_fedavg_unsw(concepts, test_set, device, verbose=False),
            'nonstat':    lambda: run_nonstationary_fedavg_unsw(concepts, test_set, device, verbose=False),
            'rehearsal':  lambda: run_cda_fedavg_unsw(concepts, test_set, device, verbose=False, always_adapt=True),
            'cda':        lambda: run_cda_fedavg_unsw(concepts, test_set, device, verbose=False,
                                                        detector_mode=DETECTOR, fusion=FUSION),
        }

    out = {}
    for name in configs:
        # same starting RNG state for every configuration
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        res = runners[name]()
        results = res[1]
        row = _row(results)
        if len(res) > 2:                      # CDA / rehearsal also return drifts
            row['detected'] = res[2]
        out[name] = row
        print(f"    {name:<12} {time.time()-t0:>5.0f}s | acc={row['overall']:.4f} "
              f"macroF1={row['macro_f1']:.4f}"
              + (f" | drifts={len(row['detected'])}/4 rounds={row['round']}"
                 if 'detected' in row else ""), flush=True)
    return out


def summarise(results, dsets, SEEDS, configs=None):
    configs = configs or CONFIGS
    print(f"\n{'='*78}\nMEAN ± STD OVER {len(SEEDS)} SEEDS (α={ALPHA})\n{'='*78}")
    lines = []
    for which in dsets:
        print(f"\n{which.upper()}")
        print(f"  {'config':<14}{'accuracy':>18}{'macro-F1':>18}")
        print("  " + "-" * 48)
        for name in configs:
            accs = np.array([results[which][str(s)][name]['overall'] for s in SEEDS])
            f1s = np.array([results[which][str(s)][name]['macro_f1'] for s in SEEDS])
            print(f"  {name:<14}{accs.mean():>10.4f}±{accs.std():<7.4f}"
                  f"{f1s.mean():>10.4f}±{f1s.std():<7.4f}")
            lines.append((which, name, accs.mean(), accs.std(), f1s.mean(), f1s.std()))

        def f1s_of(n):
            return np.array([results[which][str(s)][n]['macro_f1'] for s in SEEDS])

        print("\n  attributable differences (macro-F1):")
        for label, a, b in [
                ("forgetting cost   (A2 − B)", 'iid_stream', 'nonstat'),
                ("rehearsal value   (B2 − B)", 'rehearsal', 'nonstat'),
                ("detector value    (C − B2)", 'cda', 'rehearsal'),
                ("published gap     (A − B) ", 'iid_pooled', 'nonstat'),
                ("published recovery(C − B) ", 'cda', 'nonstat')]:
            if a not in configs or b not in configs:
                continue          # subset run — that attribution isn't available
            d = f1s_of(a) - f1s_of(b)
            print(f"    {label}  {d.mean():+.4f} ± {d.std():.4f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--dataset', choices=['cicids', 'unsw', 'both'], default='both')
    ap.add_argument('--seeds', type=int, default=len(SEEDS))
    ap.add_argument('--configs', default=','.join(CONFIGS),
                    help='comma-separated subset of ' + ','.join(CONFIGS))
    ap.add_argument('--out', default='baseline_v2_alpha05.json')
    ap.add_argument('--memory-rule', choices=['cap', 'floor'], default='cap')
    ap.add_argument('--concept-mode', choices=['5day', '8session'], default=None,
                    help="CICIDS2017 granularity; 8session gives 6 drifts + 1 non-drift")
    ap.add_argument('--data-version', choices=['original', 'improved'], default=None,
                    help="CICIDS2017 source: 'improved' = Engelen et al. WTMC 2021")
    ap.add_argument('--norm', choices=['batch', 'layer'], default=None,
                    help='default: whatever ids_loader.NORM is set to')
    ap.add_argument('--optimizer', choices=['sgd', 'momentum', 'adam', 'adamw'],
                    default=None, help='default: pipeline OPTIMIZER')
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--detector', choices=['conf', 'recon', 'hybrid'], default=None,
                    help='default: module DETECTOR')
    ap.add_argument('--fusion', choices=['or', 'and'], default='or')
    ap.add_argument('--quorum', type=int, default=5,
                    help='clients that must fire; 5 from null_control (0-8%% FP)')
    ap.add_argument('--local-norm', action='store_true',
                    help='FedBN: keep normalisation layers local (never averaged)')
    args = ap.parse_args()

    seeds = SEEDS[:args.seeds]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_pc = 500 if args.quick else 5000
    dsets = ['cicids', 'unsw'] if args.dataset == 'both' else [args.dataset]
    configs = [c.strip() for c in args.configs.split(',') if c.strip()]
    import ids_cda_pipeline, unsw_cda_pipeline
    ids_cda_pipeline.QUORUM = args.quorum
    unsw_cda_pipeline.QUORUM = args.quorum
    ids_cda_pipeline.MEMORY_RULE = args.memory_rule
    unsw_cda_pipeline.MEMORY_RULE = args.memory_rule
    ids_cda_pipeline.LOCAL_NORM = args.local_norm
    unsw_cda_pipeline.LOCAL_NORM = args.local_norm
    if args.data_version:
        ids_loader.DATA_VERSION = args.data_version
    if args.concept_mode:
        ids_loader.CONCEPT_MODE = args.concept_mode
        if args.concept_mode == '8session':
            ids_cda_pipeline.N_CONCEPTS = 8
            ids_cda_pipeline.R_PER_CONCEPT = 5
            ids_cda_pipeline.N_ROUNDS = 40
    if args.norm:
        ids_loader.NORM = unsw_loader.NORM = args.norm
    if args.optimizer:
        ids_cda_pipeline.OPTIMIZER = unsw_cda_pipeline.OPTIMIZER = args.optimizer
    if args.lr:
        ids_cda_pipeline.SGD_LR = unsw_cda_pipeline.SGD_LR = args.lr
    global DETECTOR, FUSION
    if args.detector:
        DETECTOR = args.detector
    FUSION = args.fusion

    print(f"{'='*78}")
    print(f"Corrected baseline — α={ALPHA}, seeds={seeds}, max_per_class={max_pc}")
    print(f"configs={configs} | memory_rule={args.memory_rule} "
          f"| detector={DETECTOR}/{args.fusion} quorum={args.quorum}")
    print(f"data={ids_loader.DATA_VERSION} concepts={ids_loader.CONCEPT_MODE} "
          f"norm={ids_loader.NORM} "
          f"optimizer={ids_cda_pipeline.OPTIMIZER} "
          f"lr={ids_cda_pipeline.SGD_LR} local_norm={args.local_norm}")
    print(f"{'='*78}", flush=True)

    results = {'meta': {'alpha': ALPHA, 'seeds': seeds, 'max_per_class': max_pc,
                        'n_clients': N_CLIENTS, 'configs': configs,
                        'memory_rule': args.memory_rule, 'norm': ids_loader.NORM,
                        'data_version': ids_loader.DATA_VERSION,
                        'optimizer': ids_cda_pipeline.OPTIMIZER,
                        'lr': ids_cda_pipeline.SGD_LR,
                        'detector': DETECTOR, 'fusion': FUSION,
                        'quorum': args.quorum,
                        'concept_mode': ids_loader.CONCEPT_MODE},
               'classes': {'cicids': CLASS_NAMES, 'unsw': CLASS_NAMES_UNSW}}
    out_path = os.path.join(CKPT_DIR, args.out)
    wall = time.time()

    for which in dsets:
        results[which] = {}
        for seed in seeds:
            print(f"\n  {which} seed={seed}", flush=True)
            ids_loader.SEED = seed
            unsw_loader.SEED = seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            if which == 'cicids':
                concepts, test_set, _, _, _ = load_ids_dataset(
                    n_clients=N_CLIENTS, alpha=ALPHA, max_per_class=max_pc,
                    test_fraction=0.2, verbose=False)
            else:
                concepts, test_set, _, _ = load_unsw_dataset(
                    n_clients=N_CLIENTS, alpha=ALPHA, max_per_class=max_pc,
                    verbose=False)

            results[which][str(seed)] = run_one(which, concepts, test_set,
                                                device, seed, configs)
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

    summarise(results, dsets, seeds, configs)
    print(f"\n[DONE] {(time.time()-wall)/60:.1f} min\n[SAVED] {out_path}")


if __name__ == '__main__':
    main()
