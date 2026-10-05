"""
partition_properties.py -- regenerates Table 4.3 (measured Dirichlet partition
properties) from the real loaders and the real dirichlet_partition, and saves it.

The table's earlier numbers were recorded only in STATE.md §3p; the script that
produced them was not kept. This one states its definitions:

  For every concept after the first, an ARRIVING class is an attack class that
  is present in that concept and in no earlier one. For each (draw, concept,
  arriving class), on the concept's 10-client split:
    holders_ge1    clients holding at least 1 sample of it
    holders_ge200  clients holding at least 200 of it. A concentration measure
                   only: the detector's minimum window (2 x Delta = 200) counts
                   confidence scores of ANY traffic, never class labels, so this
                   is not a count of clients able to detect the change (audit F2)
    top3_share     share of its samples held by the 3 largest holders
  The RARE class of a concept is its smallest arriving class; the table reports
  the rare-class values averaged over draws and concepts.
    size_ratio     largest / smallest non-empty client, per concept (median over
                   draws and concepts, because the ratio is heavy-tailed)
    size_corr      Pearson correlation of client sizes between consecutive
                   concepts, pooled over draws

The class counts come from the actual loader at seed 42 (training rows after the
5,000-per-class cap and the 20% test split); each draw re-partitions them with a
fresh generator, exactly as load_ids_dataset does per concept.

    python partition_properties.py --draws 200
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ids_loader, unsw_loader
from ids_loader import load_ids_dataset, dirichlet_partition, CKPT_DIR
from unsw_loader import load_unsw_dataset, dirichlet_partition_unsw

ALPHAS = [0.1, 0.5, 1.0, 100.0]


def concept_labels(which):
    """Training labels per concept, from the real loader (partition discarded)."""
    if which == 'cicids':
        c, *_ = load_ids_dataset(n_clients=10, alpha=0.5, max_per_class=5000,
                                 test_fraction=0.2, verbose=False)
    else:
        c, *_ = load_unsw_dataset(n_clients=10, alpha=0.5, max_per_class=5000, verbose=False)
    return [np.concatenate([y for _, y in cc['client_data'] if len(y)]) for cc in c]


def measure(which, ys, alpha, draws, seed=0):
    part = dirichlet_partition if which == 'cicids' else dirichlet_partition_unsw
    rng = np.random.default_rng(seed)
    rare, ratios, pairs = [], [], []
    seen0 = set(np.unique(ys[0]).tolist())
    for _ in range(draws):
        sizes_prev, seen = None, set(seen0)
        for t, y in enumerate(ys):
            X = np.zeros((len(y), 1), np.float32)
            parts = part(X, y, 10, alpha, rng)
            sizes = np.array([len(py) for _, py in parts])
            nz = sizes[sizes > 0]
            ratios.append(nz.max() / nz.min())
            if sizes_prev is not None:
                pairs.append((sizes_prev, sizes))
            sizes_prev = sizes
            if t == 0:
                continue
            arriving = [c for c in np.unique(y).tolist() if c not in seen]
            seen |= set(np.unique(y).tolist())
            if not arriving:
                continue
            per = []
            for c in arriving:
                cnt = np.array([int((py == c).sum()) for _, py in parts])
                per.append((cnt.sum(), (cnt >= 1).sum(), (cnt >= 200).sum(),
                            np.sort(cnt)[::-1][:3].sum() / cnt.sum()))
            rare.append(min(per)[1:])                  # smallest arriving class
    a = np.array(rare, float)
    x = np.concatenate([p[0] for p in pairs]).astype(float)
    z = np.concatenate([p[1] for p in pairs]).astype(float)
    return {'holders_ge1': float(a[:, 0].mean()), 'holders_ge200': float(a[:, 1].mean()),
            'top3_share': float(a[:, 2].mean()), 'size_ratio_median': float(np.median(ratios)),
            'size_ratio_mean': float(np.mean(ratios)), 'size_corr': float(np.corrcoef(x, z)[0, 1]),
            'n_rare_cases': len(rare)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--datasets', default='cicids,unsw')
    a = ap.parse_args()
    ids_loader.SEED = unsw_loader.SEED = 42
    out = {'meta': {'draws': a.draws, 'alphas': ALPHAS, 'definitions': __doc__}}
    for which in a.datasets.split(','):
        ys = concept_labels(which)
        out[which] = {'class_counts': [np.bincount(y).tolist() for y in ys]}
        for al in ALPHAS:
            out[which][f'{al:g}'] = measure(which, ys, al, a.draws)
            print(which, al, out[which][f'{al:g}'], flush=True)
    with open(os.path.join(CKPT_DIR, 'partition_properties.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
