"""
alpha_sweep_v2.py — does the LayerNorm buffer gain hold across heterogeneity?

The α sweep, rerun on the corrected pipeline with the compute-matched
configurations and both normalisation layers.

Configurations (iid_pooled is omitted: it costs ~65% of the runtime and is a
19x-budget ceiling rather than a fair federated baseline — the α=0.5 value is
already in baseline_v2):

  A2  iid_stream   compute-matched ceiling
  B   nonstat      ordered stream, no memory
  B2  rehearsal    buffer with an oracle detector
  C   cda          buffer + Beta-CUSUM detector

Sweeps alpha in {0.1, 0.5, 1.0, 100} x seeds x {batch, layer}.
Output: checkpoints/alpha_sweep_v2.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ids_loader, unsw_loader, ids_cda_pipeline, unsw_cda_pipeline
from ids_loader import load_ids_dataset, CLASS_NAMES, CKPT_DIR
from unsw_loader import load_unsw_dataset, CLASS_NAMES_UNSW
from baseline_v2 import run_one, N_CLIENTS

ALPHAS = [0.1, 0.5, 1.0, 100.0]
SEEDS = [42, 43, 44]
CONFIGS = ['iid_stream', 'nonstat', 'rehearsal', 'cda']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--norms', default='batch,layer')
    ap.add_argument('--seeds', type=int, default=len(SEEDS))
    ap.add_argument('--out', default='alpha_sweep_v2.json')
    ap.add_argument('--detector', choices=['conf', 'recon', 'hybrid'], default='conf')
    ap.add_argument('--configs', default=','.join(CONFIGS))
    ap.add_argument('--quorum', type=int, default=1)
    ap.add_argument('--quorum-mode', choices=['count', 'weight'], default='count')
    ap.add_argument('--quorum-weight', type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_pc = 500 if args.quick else 5000
    seeds = SEEDS[:args.seeds]
    norms = [n.strip() for n in args.norms.split(',')]
    import baseline_v2
    baseline_v2.DETECTOR = args.detector
    for m in (ids_cda_pipeline, unsw_cda_pipeline):
        m.QUORUM = args.quorum
        m.QUORUM_MODE = args.quorum_mode
        m.QUORUM_WEIGHT = args.quorum_weight
    configs = [c.strip() for c in args.configs.split(',') if c.strip()]
    globals()['CONFIGS'] = configs
    alphas = [0.5] if args.quick else ALPHAS

    print(f"{'='*78}")
    print(f"alpha sweep v2 — norms={norms} alphas={alphas} seeds={seeds} "
          f"configs={CONFIGS} detector={args.detector} quorum={args.quorum}/{args.quorum_mode}@{args.quorum_weight}")
    print(f"{'='*78}", flush=True)

    out = {'meta': {'alphas': alphas, 'seeds': seeds, 'norms': norms,
                    'configs': CONFIGS, 'detector': args.detector, 'max_per_class': max_pc,
                    'n_clients': N_CLIENTS},
           'classes': {'cicids': CLASS_NAMES, 'unsw': CLASS_NAMES_UNSW}}
    path = os.path.join(CKPT_DIR, args.out)
    wall = time.time()

    for norm in norms:
        ids_loader.NORM = norm
        unsw_loader.NORM = norm
        out[norm] = {}
        for which in ('cicids', 'unsw'):
            out[norm][which] = {}
            for alpha in alphas:
                out[norm][which][str(alpha)] = {}
                for seed in seeds:
                    print(f"\n  [{norm}] {which} alpha={alpha} seed={seed}",
                          flush=True)
                    ids_loader.SEED = seed
                    unsw_loader.SEED = seed
                    torch.manual_seed(seed); np.random.seed(seed)

                    if which == 'cicids':
                        concepts, test_set, _, _, _ = load_ids_dataset(
                            n_clients=N_CLIENTS, alpha=alpha,
                            max_per_class=max_pc, test_fraction=0.2,
                            verbose=False)
                    else:
                        concepts, test_set, _, _ = load_unsw_dataset(
                            n_clients=N_CLIENTS, alpha=alpha,
                            max_per_class=max_pc, verbose=False)

                    out[norm][which][str(alpha)][str(seed)] = run_one(
                        which, concepts, test_set, device, seed, CONFIGS)
                    with open(path, 'w') as f:
                        json.dump(out, f, indent=2)

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*78}\nMACRO-F1, mean ± std over {len(seeds)} seeds\n{'='*78}")
    for which in ('cicids', 'unsw'):
        print(f"\n{which.upper()}")
        hdr = f"  {'norm':<7}{'alpha':>7}" + "".join(f"{c:>20}" for c in CONFIGS)
        print(hdr + f"{'buffer B2-B':>16}{'drifts':>9}")
        print("  " + "-" * (len(hdr) + 22))
        for norm in norms:
            for alpha in alphas:
                cell = out[norm][which][str(alpha)]
                row = f"  {norm:<7}{alpha:>7g}"
                vals = {}
                for c in CONFIGS:
                    v = np.array([cell[str(s)][c]['macro_f1'] for s in seeds])
                    vals[c] = v
                    row += f"{v.mean():>13.4f}±{v.std():<6.4f}"
                d = vals['rehearsal'] - vals['nonstat']
                nd = np.mean([len(cell[str(s)]['cda'].get('detected', []))
                              for s in seeds])
                row += f"{d.mean():>+11.4f}±{d.std():<4.3f}{nd:>8.1f}/4"
                print(row)

    print(f"\n[DONE] {(time.time()-wall)/60:.1f} min\n[SAVED] {path}")


if __name__ == '__main__':
    main()
