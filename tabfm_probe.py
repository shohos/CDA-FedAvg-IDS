"""
tabfm_probe.py — what does TabFM actually give us, before any FL integration?

Three questions, in the order that decides whether integration is worth doing:

  1. FEASIBILITY. TabFM is in-context: fit() stores a context, predict() attends
     over it. Our data is far larger than any usable context, so the binding
     constraint is how many rows we can afford to pass and what that costs in
     latency. Measured here, not assumed.

  2. VALUE. Zero-shot macro-F1 against the ceilings this project already
     established with a TRAINED model at 19x federated compute:
         CICIDS2017 (corrected)  0.9870
         UNSW-NB15               0.4138
     UNSW is the interesting one: four of its ten classes score ~0 at that
     ceiling, and we concluded they are unlearnable FROM THESE FEATURES. A
     foundation model with a different inductive bias is the one thing that
     could move them. If it cannot, that conclusion hardens.

  3. INTEGRATION PATH. TabFMClassifier exposes only fit/predict/predict_proba —
     there is no transform() or embedding API — so "frozen extractor + federated
     head" needs a hook into the underlying module, not the sklearn wrapper.
     This probes whether such a hook exists.

Deliberately NOT federated yet. Establish the single-machine numbers first, or we
repeat the original paper's mistake of reporting a mechanism whose value was
never isolated.

Out: checkpoints/tabfm_probe.json
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ids_loader, unsw_loader
from ids_loader import load_ids_dataset, CLASS_NAMES, CKPT_DIR
from unsw_loader import load_unsw_dataset, CLASS_NAMES_UNSW
from sklearn.metrics import f1_score, accuracy_score

CEILING = {'cicids': 0.9870, 'unsw': 0.4138}   # trained, 19x federated compute


def pooled(loader, seed=42):
    """Concepts pooled back together: the fairest single-machine view of the data."""
    ids_loader.SEED = unsw_loader.SEED = seed
    concepts, (Xte, yte), *_ = loader()
    X = np.concatenate([np.concatenate([xk for xk, _ in c['client_data'] if len(xk)])
                        for c in concepts])
    y = np.concatenate([np.concatenate([yk for _, yk in c['client_data'] if len(yk)])
                        for c in concepts])
    return X, y, Xte, yte


def stratified(X, y, n, rng):
    """CONTEXT sampling only. Balances classes so no class is absent from the
    context (an absent class can never be predicted). Never use for the test
    set: a balanced test set inflates macro-F1."""
    out = []
    classes = np.unique(y)
    per = max(1, n // len(classes))
    for c in classes:
        idx = np.where(y == c)[0]
        out.append(rng.choice(idx, min(len(idx), per), replace=False))
    out = np.concatenate(out)
    return out[rng.permutation(len(out))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['cicids', 'unsw', 'both'], default='both')
    ap.add_argument('--context', default='200,1000,4000',
                    help='context sizes to sweep')
    ap.add_argument('--test-n', type=int, default=3000)
    ap.add_argument('--estimators', type=int, default=4,
                    help='TabFM default is 32; each is a full pass, so this is '
                         'the main latency knob')
    ap.add_argument('--out', default='tabfm_probe.json')
    args = ap.parse_args()

    import torch
    from tabfm import TabFMClassifier
    from tabfm.src.pytorch import tabfm_v1_0_0 as M

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    model = M.load('classification', device=dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"TabFM loaded in {time.time()-t0:.0f}s on {dev} | {npar:,} parameters "
          f"({npar*2/1e6:.1f} MB bf16)")
    print(f"max_classes = {getattr(model, 'max_classes', '?')}  "
          f"(class_shift covers datasets with more)\n")

    res = {'meta': {'params': int(npar), 'device': dev,
                    'n_estimators': args.estimators, 'test_n': args.test_n},
           'ceilings': CEILING, 'runs': {}}

    dsets = ['cicids', 'unsw'] if args.dataset == 'both' else [args.dataset]
    for which in dsets:
        loader = (lambda: load_ids_dataset(n_clients=10, alpha=0.5,
                                           max_per_class=5000, verbose=False)) \
            if which == 'cicids' else \
            (lambda: load_unsw_dataset(n_clients=10, alpha=0.5,
                                       max_per_class=5000, verbose=False))
        names = CLASS_NAMES if which == 'cicids' else CLASS_NAMES_UNSW
        X, y, Xte, yte = pooled(loader)
        rng = np.random.default_rng(42)
        # NATURAL class proportions — a stratified/balanced test set inflates
        # macro-F1 and would not be comparable to the ceilings above, which are
        # measured on the full imbalanced test set.
        te = rng.choice(len(Xte), min(args.test_n, len(Xte)), replace=False)
        Xt, yt = Xte[te], yte[te]
        print(f"{'='*70}\n{which.upper()}  train pool {X.shape}  test {Xt.shape}  "
              f"classes {len(np.unique(y))}\n{'='*70}")
        res['runs'][which] = []

        for n_ctx in [int(v) for v in args.context.split(',')]:
            ctx = stratified(X, y, n_ctx, rng)
            clf = TabFMClassifier(model=model, n_estimators=args.estimators,
                                  random_state=42)
            t = time.time()
            clf.fit(X[ctx], y[ctx])
            t_fit = time.time() - t
            t = time.time()
            pred = clf.predict(Xt)
            t_pred = time.time() - t

            mf1 = f1_score(yt, pred, average='macro', zero_division=0)
            acc = accuracy_score(yt, pred)
            per = f1_score(yt, pred, average=None, labels=range(len(names)),
                           zero_division=0)
            print(f"  context {len(ctx):>5}  macro-F1 {mf1:.4f}  acc {acc:.4f}  "
                  f"| fit {t_fit:.1f}s predict {t_pred:.1f}s "
                  f"({1000*t_pred/len(Xt):.1f} ms/row)")
            print(f"      per-class: " + "  ".join(
                f"{c}={v:.2f}" for c, v in zip(names, per)))
            res['runs'][which].append(
                {'context': int(len(ctx)), 'macro_f1': float(mf1),
                 'accuracy': float(acc), 'fit_s': t_fit, 'predict_s': t_pred,
                 'per_class_f1': {c: float(v) for c, v in zip(names, per)}})

        best = max(r['macro_f1'] for r in res['runs'][which])
        print(f"\n  best zero-shot {best:.4f}  vs trained ceiling "
              f"{CEILING[which]:.4f}  ({best - CEILING[which]:+.4f})\n")

    path = os.path.join(CKPT_DIR, args.out)
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"[SAVED] {path}")


if __name__ == '__main__':
    main()
