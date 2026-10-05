"""
opt_sweep.py — the optimiser was inherited too. What does a tuned one buy?

lr_sweep.py covers SGD across five learning rates for both normalisations. This
adds the optimisers that were never tried: SGD with momentum, and Adam. Each gets
its own learning-rate grid, because comparing Adam at 0.01 against SGD at 0.01 is
not a comparison of optimisers.

Two questions, and they are separate:

  1. diagnostic — does BatchNorm's +0.31 deficit survive a properly tuned
     optimiser? If a tuned BatchNorm matches LayerNorm, the normalisation result
     was a tuning artefact.
  2. improvement — does the best (optimiser, lr, norm) beat our current
     0.8236 on CICIDS? The baseline is fixed and reproducible; improving on it is
     the point now.

FL caveat, documented in the pipelines: a fresh model is built per local_train
call, so client optimiser state does not persist across rounds. Adam's moments
reset every round and only adapt within a round's 10 local epochs. That is the
known limitation motivating server-side adaptive aggregation (Reddi et al.,
"Adaptive Federated Optimization", ICLR 2021), which is the natural follow-up if
client-side Adam underdelivers here.

Output: checkpoints/opt_sweep.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ids_loader, unsw_loader, ids_cda_pipeline as P, unsw_cda_pipeline as U
from ids_loader import load_ids_dataset, CKPT_DIR
from unsw_loader import load_unsw_dataset

GRID = {'momentum': [0.003, 0.01, 0.03],
        'adam':     [3e-4, 1e-3, 3e-3]}
SEEDS = [42, 43, 44]
ALPHA = 0.5


def run_one(which, norm, opt, lr, seed, device):
    ids_loader.SEED = unsw_loader.SEED = seed
    ids_loader.NORM = unsw_loader.NORM = norm
    P.OPTIMIZER = U.OPTIMIZER = opt
    P.SGD_LR = U.SGD_LR = lr
    torch.manual_seed(seed); np.random.seed(seed)
    if which == 'cicids':
        concepts, test_set, _, _, _ = load_ids_dataset(
            n_clients=P.N_CLIENTS, alpha=ALPHA, max_per_class=5000,
            test_fraction=0.2, verbose=False)
        _, res, _ = P.run_cda_fedavg_ids(concepts, test_set, device,
                                         verbose=False, always_adapt=True)
    else:
        concepts, test_set, _, _ = load_unsw_dataset(
            n_clients=U.N_CLIENTS, alpha=ALPHA, max_per_class=5000, verbose=False)
        _, res, _ = U.run_cda_fedavg_unsw(concepts, test_set, device,
                                          verbose=False, always_adapt=True)
    return res[-1]['macro_f1']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['cicids', 'unsw', 'both'], default='both')
    ap.add_argument('--seeds', type=int, default=len(SEEDS))
    ap.add_argument('--out', default='opt_sweep.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = SEEDS[:args.seeds]
    dsets = ('cicids', 'unsw') if args.dataset == 'both' else (args.dataset,)
    out = {'meta': {'grid': GRID, 'seeds': seeds, 'alpha': ALPHA,
                    'arm': 'rehearsal (oracle detector)',
                    'note': 'SGD rows are in lr_sweep.json'}}
    print(f"optimiser sweep — {GRID} seeds={seeds}\n", flush=True)
    wall = time.time()

    for which in dsets:
        out[which] = {}
        for norm in ('batch', 'layer'):
            out[which][norm] = {}
            for opt, lrs in GRID.items():
                for lr in lrs:
                    vals = [run_one(which, norm, opt, lr, s, device) for s in seeds]
                    out[which][norm][f"{opt}@{lr}"] = [float(v) for v in vals]
                    print(f"  {which:<7}{norm:<7}{opt:<9}lr={lr:<8}"
                          f"macroF1={np.mean(vals):.4f} ± {np.std(vals):.4f}",
                          flush=True)
                    with open(os.path.join(CKPT_DIR, args.out), 'w') as f:
                        json.dump(out, f, indent=2)

    P.OPTIMIZER = U.OPTIMIZER = 'sgd'
    P.SGD_LR = U.SGD_LR = 0.01

    print(f"\n{'='*72}\nBEST CONFIGURATION PER NORMALISATION\n{'='*72}")
    for which in dsets:
        print(f"\n{which.upper()}")
        for norm in ('batch', 'layer'):
            rows = sorted(((np.mean(v), k) for k, v in out[which][norm].items()),
                          reverse=True)
            best_v, best_k = rows[0]
            print(f"  {norm:<7} best {best_k:<16} {best_v:.4f}   "
                  f"(worst {rows[-1][1]} {rows[-1][0]:.4f})")
    print(f"\n[DONE] {(time.time()-wall)/60:.1f} min")
    print(f"[SAVED] {os.path.join(CKPT_DIR, args.out)}")
    print("\nCompare against SGD@0.01 baselines: CICIDS batch 0.5005 / layer 0.8236")


if __name__ == '__main__':
    main()
