"""
lr_sweep.py — is the "LayerNorm gain" actually an untuned learning rate?

Five explanations for BatchNorm's +0.31 deficit in the rehearsal arm have been
tested and refuted:

  federated statistic averaging  FedBN recovers 7%
  tabular inductive bias         IID-pooled gains only +0.010
  inference statistics mismatch  batch-stat evaluation recovers 7%
  generalisation failure         BN's test drop is -0.0001; it fails on its own buffer
  aggregation destroying the fit averaging HELPS both norms equally (+0.034)

The gap is present in a single client's local training, before any aggregation.
That leaves the optimiser. BatchNorm rescales gradient magnitudes and normally
tolerates — often needs — a larger learning rate than an unnormalised or
layer-normalised network. Our lr = 0.01 with no momentum was inherited from
Casado's HAR CNN and has never been re-tuned for either normalisation.

If BatchNorm at a better learning rate closes on LayerNorm, then the headline
"+0.31 from a normalisation swap" is really "the inherited learning rate suited
LayerNorm and starved BatchNorm" — a much weaker claim that must be reported as
such. If BatchNorm stays behind at every learning rate, the normalisation result
stands and is now well-defended.

Output: checkpoints/lr_sweep.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ids_loader, unsw_loader, ids_cda_pipeline as P, unsw_cda_pipeline as U
from ids_loader import load_ids_dataset, CKPT_DIR
from unsw_loader import load_unsw_dataset

LRS = [0.003, 0.01, 0.03, 0.1, 0.3]
SEEDS = [42, 43, 44]
ALPHA = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['cicids', 'unsw', 'both'], default='both')
    ap.add_argument('--seeds', type=int, default=len(SEEDS))
    ap.add_argument('--out', default='lr_sweep.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = SEEDS[:args.seeds]
    dsets = ('cicids', 'unsw') if args.dataset == 'both' else (args.dataset,)
    out = {'meta': {'lrs': LRS, 'seeds': seeds, 'alpha': ALPHA,
                    'arm': 'rehearsal (oracle detector)'}}
    print(f"learning-rate sweep — lrs={LRS} seeds={seeds} alpha={ALPHA}")
    print(f"inherited default is 0.01\n", flush=True)
    wall = time.time()

    for which in dsets:
        out[which] = {}
        for norm in ('batch', 'layer'):
            out[which][norm] = {}
            for lr in LRS:
                vals = []
                for seed in seeds:
                    ids_loader.SEED = unsw_loader.SEED = seed
                    ids_loader.NORM = unsw_loader.NORM = norm
                    P.SGD_LR = U.SGD_LR = lr
                    torch.manual_seed(seed); np.random.seed(seed)
                    if which == 'cicids':
                        concepts, test_set, _, _, _ = load_ids_dataset(
                            n_clients=P.N_CLIENTS, alpha=ALPHA,
                            max_per_class=5000, test_fraction=0.2, verbose=False)
                        _, res, _ = P.run_cda_fedavg_ids(
                            concepts, test_set, device, verbose=False,
                            always_adapt=True)
                    else:
                        concepts, test_set, _, _ = load_unsw_dataset(
                            n_clients=U.N_CLIENTS, alpha=ALPHA,
                            max_per_class=5000, verbose=False)
                        _, res, _ = U.run_cda_fedavg_unsw(
                            concepts, test_set, device, verbose=False,
                            always_adapt=True)
                    vals.append(res[-1]['macro_f1'])
                out[which][norm][str(lr)] = [float(v) for v in vals]
                print(f"  {which:<7}{norm:<7}lr={lr:<7} "
                      f"macroF1={np.mean(vals):.4f} ± {np.std(vals):.4f}", flush=True)
                with open(os.path.join(CKPT_DIR, args.out), 'w') as f:
                    json.dump(out, f, indent=2)

    P.SGD_LR = U.SGD_LR = 0.01          # restore the documented default
    print(f"\n{'='*70}\nBEST LEARNING RATE PER NORMALISATION\n{'='*70}")
    for which in dsets:
        print(f"\n{which.upper()}")
        print(f"  {'norm':<8}" + "".join(f"{lr:>12}" for lr in LRS) + f"{'best':>10}")
        best = {}
        for norm in ('batch', 'layer'):
            row = f"  {norm:<8}"
            means = []
            for lr in LRS:
                m = np.mean(out[which][norm][str(lr)])
                means.append(m)
                row += f"{m:>12.4f}"
            b = LRS[int(np.argmax(means))]
            best[norm] = (b, max(means))
            print(row + f"{b:>10}")
        gap_at_default = (np.mean(out[which]['layer']['0.01'])
                          - np.mean(out[which]['batch']['0.01']))
        gap_at_best = best['layer'][1] - best['batch'][1]
        print(f"  gap at the inherited lr=0.01 : {gap_at_default:+.4f}")
        print(f"  gap with each norm at ITS best lr: {gap_at_best:+.4f}")
        print(f"  -> {1 - gap_at_best/gap_at_default:.0%} of the reported "
              f"'LayerNorm gain' was learning rate"
              if abs(gap_at_default) > 1e-9 else "")

    print(f"\n[DONE] {(time.time()-wall)/60:.1f} min")
    print(f"[SAVED] {os.path.join(CKPT_DIR, args.out)}")


if __name__ == '__main__':
    main()
