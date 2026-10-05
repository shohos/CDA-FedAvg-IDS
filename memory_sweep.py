"""
memory_sweep.py — are we memory-bound or method-bound?

Our system sits at or above the compute-matched ceiling on both datasets, so
mechanism work has stopped paying. But the rehearsal buffer only ever holds ~20%
of the training data (L=1400 -> ~7,900 rows of 39,316), and L=1400 was inherited
from Casado's HAR setup — 7 balanced activity classes — and never re-derived for
7-10 severely imbalanced intrusion classes. Nobody has drawn the curve.

So: fix the compute budget, sweep the memory budget, and see where accuracy goes.

  * if accuracy climbs toward the 19x-budget ceiling as L grows, the limit is
    memory, and the deployment story becomes "accuracy per KB of buffer" — a
    trade-off curve that is nowhere near saturated
  * if it plateaus well below, the limit really is the method, and representation
    learning is the answer

Uses the `rehearsal` arm (oracle detector) so the buffer is measured without the
detector confounding it, with `nonstat` as the no-memory reference.

The unbounded arm sets PER_CLASS effectively to infinity: the buffer becomes every
sample the client has seen, which is the upper bound rehearsal can possibly reach.

Output: checkpoints/memory_sweep.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ids_loader, unsw_loader, ids_cda_pipeline, unsw_cda_pipeline
from ids_loader import load_ids_dataset, CLASS_NAMES, CKPT_DIR
from unsw_loader import load_unsw_dataset, CLASS_NAMES_UNSW
from baseline_v2 import run_one, N_CLIENTS

L_VALUES = [350, 700, 1400, 2800, 5600, None]      # None = unbounded
SEEDS = [42, 43, 44]
ALPHA = 0.5
UNBOUNDED = 10 ** 9


def set_budget(L):
    """Set L and the derived per-class quota on both pipelines."""
    for mod, n_cls in ((ids_cda_pipeline, ids_cda_pipeline.N_CLASSES),
                       (unsw_cda_pipeline, unsw_cda_pipeline.N_CLASSES_UNSW)):
        if L is None:
            mod.L_CAPACITY = None
            mod.PER_CLASS = UNBOUNDED
        else:
            mod.L_CAPACITY = L
            mod.PER_CLASS = L // (2 * n_cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=len(SEEDS))
    ap.add_argument('--norm', choices=['batch', 'layer'], default='layer')
    ap.add_argument('--out', default='memory_sweep.json')
    ap.add_argument('--alpha', type=float, default=ALPHA)
    ap.add_argument('--dataset', choices=['cicids', 'unsw', 'both'], default='both')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_pc = 500 if args.quick else 5000
    seeds = SEEDS[:args.seeds]
    Ls = [1400, None] if args.quick else L_VALUES
    ids_loader.NORM = unsw_loader.NORM = args.norm
    globals()['ALPHA'] = args.alpha
    dsets = ('cicids', 'unsw') if args.dataset == 'both' else (args.dataset,)

    print(f"{'='*78}")
    print(f"memory sweep — L={Ls} norm={args.norm} seeds={seeds} alpha={ALPHA}")
    print(f"compute budget FIXED at 25 rounds x 10 epochs throughout")
    print(f"{'='*78}", flush=True)

    out = {'meta': {'L_values': [str(L) for L in Ls], 'seeds': seeds,
                    'alpha': ALPHA, 'norm': args.norm, 'max_per_class': max_pc}}
    path = os.path.join(CKPT_DIR, args.out)
    wall = time.time()

    for which in dsets:
        out[which] = {}
        n_feat = (ids_loader.N_FEATURES if which == 'cicids'
                  else unsw_loader.N_FEATURES_UNSW)
        for L in Ls:
            key = 'unbounded' if L is None else str(L)
            out[which][key] = {}
            for seed in seeds:
                ids_loader.SEED = unsw_loader.SEED = seed
                torch.manual_seed(seed); np.random.seed(seed)
                set_budget(L)

                if which == 'cicids':
                    concepts, test_set, _, _, _ = load_ids_dataset(
                        n_clients=N_CLIENTS, alpha=ALPHA, max_per_class=max_pc,
                        test_fraction=0.2, verbose=False)
                else:
                    concepts, test_set, _, _ = load_unsw_dataset(
                        n_clients=N_CLIENTS, alpha=ALPHA, max_per_class=max_pc,
                        verbose=False)
                total_rows = sum(len(Xk) for c in concepts
                                 for Xk, _ in c['client_data'])
                print(f"\n  {which} L={key} seed={seed} "
                      f"(per_class={'inf' if L is None else L//(2*(7 if which=='cicids' else 10))})",
                      flush=True)
                cfgs = ['nonstat', 'rehearsal'] if L == Ls[0] else ['rehearsal']
                r = run_one(which, concepts, test_set, device, seed, cfgs)
                buf = None
                det = r['rehearsal'].get('detected') or []
                if det:
                    buf = det[-1].get('memory_client0')
                r['_buffer_client0'] = buf
                r['_total_rows'] = total_rows
                r['_kb_client0'] = (buf * n_feat * 4 / 1024) if buf else None
                out[which][key][str(seed)] = r
                with open(path, 'w') as f:
                    json.dump(out, f, indent=2)

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*78}\nMEMORY BUDGET vs ACCURACY (rehearsal arm, "
          f"{len(seeds)} seeds)\n{'='*78}")
    for which in ('cicids', 'unsw'):
        print(f"\n{which.upper()}")
        print(f"  {'L':>10}{'per-class':>11}{'buffer(c0)':>12}{'KB(c0)':>9}"
              f"{'macro-F1':>18}")
        print("  " + "-" * 60)
        base = None
        for L in Ls:
            key = 'unbounded' if L is None else str(L)
            cell = out[which][key]
            f1 = np.array([cell[str(s)]['rehearsal']['macro_f1'] for s in seeds])
            bufs = [cell[str(s)]['_buffer_client0'] for s in seeds
                    if cell[str(s)]['_buffer_client0']]
            kbs = [cell[str(s)]['_kb_client0'] for s in seeds
                   if cell[str(s)]['_kb_client0']]
            tot = np.mean([cell[str(s)]['_total_rows'] for s in seeds])
            pc = ('inf' if L is None
                  else str(L // (2 * (7 if which == 'cicids' else 10))))
            b = f"{np.mean(bufs):.0f}" if bufs else "—"
            k = f"{np.mean(kbs):.0f}" if kbs else "—"
            print(f"  {key:>10}{pc:>11}{b:>12}{k:>9}"
                  f"{f1.mean():>11.4f}±{f1.std():<6.4f}")
        ns = np.array([out[which][str(Ls[0])][str(s)]['nonstat']['macro_f1']
                       for s in seeds])
        print(f"  {'no memory':>10}{'—':>11}{'0':>12}{'0':>9}"
              f"{ns.mean():>11.4f}±{ns.std():<6.4f}")

    print(f"\n[DONE] {(time.time()-wall)/60:.1f} min\n[SAVED] {path}")


if __name__ == '__main__':
    main()
