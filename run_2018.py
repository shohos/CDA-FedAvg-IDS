"""
run_2018.py — the whole protocol on CSE-CIC-IDS2018.

Generalisation test: does the configuration tuned on CICIDS2017 transfer to a
different capture, a different attack schedule, and a longer drift sequence
(6 transitions vs 4)?

Reuses IDSDNN and every run_* function unchanged — 2018 also yields 78 features
and 7 classes after preprocessing. Only the loader and the round count differ:
7 concepts x 5 rounds = 35, preserving R_PER_CONCEPT = 5.

CLASS_NAMES is rebound on both modules so per-class keys carry 2018's class names
rather than 2017's. The functions resolve it as a module global at call time, so
rebinding works — the same late-binding property that bit us four times as a bug
is useful here.

Output: checkpoints/ids2018_results.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ids_loader, ids_cda_pipeline as P
from ids_loader import CKPT_DIR
from ids2018_loader import (load_ids2018_dataset, CLASS_NAMES_2018,
                            N_CONCEPTS_2018, N_CONCEPTS_2018_DAILY,
                            CONCEPTS_2018_DAILY, TRUE_DRIFT_IDX,
                            TRUE_NONDRIFT_IDX)
import baseline_v2

SEEDS = [42, 43, 44]
ALPHA = 0.5
CONFIGS = ['iid_stream', 'nonstat', 'rehearsal', 'cda']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=len(SEEDS))
    ap.add_argument('--max-per-class', type=int, default=5000)
    ap.add_argument('--quorum', type=int, default=5)
    ap.add_argument('--detector', default='hybrid')
    ap.add_argument('--out', default='ids2018_results.json')
    ap.add_argument('--concepts', choices=['daily', 'phase'], default='daily')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = SEEDS[:args.seeds]

    # 7 concepts instead of 5; keep 5 rounds per concept
    NC = N_CONCEPTS_2018_DAILY if args.concepts == 'daily' else N_CONCEPTS_2018
    P.N_CONCEPTS = NC
    P.R_PER_CONCEPT = 5
    P.N_ROUNDS = 5 * NC
    P.QUORUM = args.quorum
    P.CLASS_NAMES = CLASS_NAMES_2018
    ids_loader.CLASS_NAMES = CLASS_NAMES_2018      # evaluate_ids reads this
    baseline_v2.DETECTOR = args.detector

    print(f"{'='*78}")
    print(f"CSE-CIC-IDS2018 [{args.concepts}] — {NC} concepts x 5 rounds "
          f"= {P.N_ROUNDS} rounds, α={ALPHA}, seeds={seeds}")
    print(f"norm={ids_loader.NORM} optimizer={P.OPTIMIZER} lr={P.SGD_LR} "
          f"detector={args.detector} quorum={args.quorum}")
    print(f"{'='*78}", flush=True)

    out = {'meta': {'seeds': seeds, 'alpha': ALPHA, 'concepts': NC,
                    'rounds': P.N_ROUNDS, 'norm': ids_loader.NORM,
                    'optimizer': P.OPTIMIZER, 'lr': P.SGD_LR,
                    'detector': args.detector, 'quorum': args.quorum,
                    'max_per_class': args.max_per_class},
           'classes': CLASS_NAMES_2018}
    path = os.path.join(CKPT_DIR, args.out)
    wall = time.time()

    for seed in seeds:
        print(f"\n  seed={seed}", flush=True)
        ids_loader.SEED = seed
        torch.manual_seed(seed); np.random.seed(seed)
        concepts, test_set, _, _, _ = load_ids2018_dataset(
            n_clients=P.N_CLIENTS, alpha=ALPHA,
            max_per_class=args.max_per_class, verbose=False,
            mode=args.concepts)
        out[str(seed)] = baseline_v2.run_one('cicids', concepts, test_set,
                                             device, seed, CONFIGS)
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

    print(f"\n{'='*78}\nMEAN ± STD OVER {len(seeds)} SEEDS\n{'='*78}")
    print(f"  {'config':<14}{'accuracy':>18}{'macro-F1':>18}")
    for c in CONFIGS:
        a = np.array([out[str(s)][c]['overall'] for s in seeds])
        f = np.array([out[str(s)][c]['macro_f1'] for s in seeds])
        print(f"  {c:<14}{a.mean():>10.4f}±{a.std():<7.4f}{f.mean():>10.4f}±{f.std():<7.4f}")

    def f1s(n):
        return np.array([out[str(s)][n]['macro_f1'] for s in seeds])
    print("\n  attributable differences (macro-F1):")
    for lbl, x, y in [("forgetting cost (A2-B)", 'iid_stream', 'nonstat'),
                      ("rehearsal value (B2-B)", 'rehearsal', 'nonstat'),
                      ("detector value  (C-B2)", 'cda', 'rehearsal')]:
        d = f1s(x) - f1s(y)
        print(f"    {lbl}  {d.mean():+.4f} ± {d.std():.4f}")
    nd = [len(out[str(s)]['cda'].get('detected', [])) for s in seeds]
    print(f"    adaptations fired: {nd} of {NC - 1} transitions")
    if args.concepts == 'daily':
        names = [c[0] for c in CONCEPTS_2018_DAILY]
        tp = fp = 0
        for s in seeds:
            fired = {d['concept'] for d in out[str(s)]['cda'].get('detected', [])}
            for i, n in enumerate(names):
                if i == 0:
                    continue
                if n in fired:
                    tp += i in TRUE_DRIFT_IDX
                    fp += i in TRUE_NONDRIFT_IDX
        nseed = len(seeds)
        print("")
        print("  detection on real data (new class on days 2,4,6,8,10; "
              "days 3,5,7,9 repeat the previous class):")
        print(f"    true positives  {tp/nseed:.1f} of {len(TRUE_DRIFT_IDX)}")
        print(f"    false positives {fp/nseed:.1f} of {len(TRUE_NONDRIFT_IDX)}")
        for i, n in enumerate(names):
            if i == 0:
                continue
            hits = sum(n in {d['concept'] for d in out[str(s)]['cda'].get('detected', [])}
                       for s in seeds)
            kind = 'DRIFT   ' if i in TRUE_DRIFT_IDX else 'no-drift'
            print(f"      {n:<24}{kind}  fired {hits}/{nseed}")

    print(f"\n  per-class F1 (CDA arm)")
    for i, c in enumerate(CLASS_NAMES_2018):
        v = np.mean([out[str(s)]['cda'].get('f1_' + c) or 0.0 for s in seeds])
        print(f"    {c:<14}{v:.3f}")

    print(f"\n[DONE] {(time.time()-wall)/60:.1f} min\n[SAVED] {path}")


if __name__ == '__main__':
    main()
