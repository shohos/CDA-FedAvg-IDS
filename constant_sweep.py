"""
constant_sweep.py — sweep the remaining inherited constants.

Four constants inherited from Casado's HAR setup have now been swept, and three
of the four mattered:

  compute budget (25 rounds)   made the IID ceiling look 19x better
  memory budget  (L=1400)      worth +0.08, and a curve nobody had drawn
  learning rate  (0.01)        worth +0.08, and inflated the normalisation result
  PER_CLASS = L/(2M)           inverted relative to the source paper

These are still untouched, all from the same source: BATCH_SIZE=100,
LOCAL_EPOCHS=10, CLIP_NORM=10.0. Each is swept one at a time from the current best
configuration (LayerNorm, SGD-momentum @ 0.01, L=1400) on the rehearsal arm, which
is the arm that carries the result.

LOCAL_EPOCHS changes the compute budget, so its column reports relative gradient
steps alongside macro-F1 — a gain bought with 2x the compute is not the same kind
of gain as one bought for free.

Note: E and B could not be overridden before this script — like SGD_LR and SEED
before them, they were bound at function-definition time.

Output: checkpoints/constant_sweep.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ids_loader, unsw_loader, ids_cda_pipeline as P, unsw_cda_pipeline as U
from ids_loader import load_ids_dataset, CKPT_DIR
from unsw_loader import load_unsw_dataset

SWEEPS = {
    'BATCH_SIZE':   [25, 50, 100, 200, 400],
    'LOCAL_EPOCHS': [2, 5, 10, 20],
    'CLIP_NORM':    [1.0, 5.0, 10.0, 1e9],      # 1e9 = effectively off
}
DEFAULTS = {'BATCH_SIZE': 100, 'LOCAL_EPOCHS': 10, 'CLIP_NORM': 10.0}
SEEDS = [42, 43, 44]
ALPHA = 0.5


def run(which, seed, device):
    ids_loader.SEED = unsw_loader.SEED = seed
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
    ap.add_argument('--out', default='constant_sweep.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = SEEDS[:args.seeds]
    dsets = ('cicids', 'unsw') if args.dataset == 'both' else (args.dataset,)
    out = {'meta': {'sweeps': SWEEPS, 'defaults': DEFAULTS, 'seeds': seeds,
                    'norm': ids_loader.NORM, 'optimizer': P.OPTIMIZER,
                    'lr': P.SGD_LR, 'arm': 'rehearsal'}}
    print(f"constant sweep — from best config "
          f"(norm={ids_loader.NORM}, opt={P.OPTIMIZER}, lr={P.SGD_LR})")
    print(f"one constant at a time; others held at {DEFAULTS}\n", flush=True)
    wall = time.time()

    for which in dsets:
        out[which] = {}
        for const, values in SWEEPS.items():
            out[which][const] = {}
            for v in values:
                for mod in (P, U):                    # reset, then set one
                    for k, d in DEFAULTS.items():
                        setattr(mod, k, d)
                    setattr(mod, const, v)
                vals = [run(which, s, device) for s in seeds]
                out[which][const][str(v)] = [float(x) for x in vals]
                tag = 'off' if const == 'CLIP_NORM' and v > 1e8 else str(v)
                mark = '  (default)' if v == DEFAULTS[const] else ''
                print(f"  {which:<7}{const:<14}{tag:<8}"
                      f"macroF1={np.mean(vals):.4f} ± {np.std(vals):.4f}{mark}",
                      flush=True)
                with open(os.path.join(CKPT_DIR, args.out), 'w') as f:
                    json.dump(out, f, indent=2)
        for mod in (P, U):                            # restore
            for k, d in DEFAULTS.items():
                setattr(mod, k, d)

    print(f"\n{'='*74}\nBEST VALUE PER CONSTANT (vs the inherited default)\n{'='*74}")
    for which in dsets:
        print(f"\n{which.upper()}")
        for const, values in SWEEPS.items():
            means = {v: np.mean(out[which][const][str(v)]) for v in values}
            best = max(means, key=means.get)
            d = DEFAULTS[const]
            btag = 'off' if const == 'CLIP_NORM' and best > 1e8 else best
            note = ''
            if const == 'LOCAL_EPOCHS':
                note = f"  [{best/d:.1f}x compute]"
            print(f"  {const:<14}default {d:<8}{means[d]:.4f}   "
                  f"best {str(btag):<8}{means[best]:.4f}   "
                  f"{means[best]-means[d]:+.4f}{note}")

    print(f"\n[DONE] {(time.time()-wall)/60:.1f} min")
    print(f"[SAVED] {os.path.join(CKPT_DIR, args.out)}")


if __name__ == '__main__':
    main()
