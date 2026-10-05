"""
analyse_agg.py -- protocol v3 (aggregator study): every rule against FedAvg,
paired by seed, for each arm, dataset and alpha, with Holm-adjusted p-values over
the primary family (rehearsal arm, alpha 0.1 and 100, all datasets).

    python analyse_agg.py [--dir agg] [--out agg_analysis.json]
"""
import argparse, json, os
import numpy as np
from analyse_final_eval import interval, forgetting, forward_mean, peak_memory, holm, CKPT

RULES = ['fedavg', 'fedprox', 'scaffold', 'median']
ARMS = ['nonstat', 'rehearsal', 'cda']
DATASETS = ['cicids', 'unsw', 'ids2018']
PRIMARY_ALPHAS = ('0.1', '100')


def load(d, rule, ds):
    p = os.path.join(CKPT, d, f'{rule}_{ds}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='agg')
    ap.add_argument('--out', default='agg_analysis.json')
    a = ap.parse_args()
    out = {}
    for ds in DATASETS:
        files = {r: load(a.dir, r, ds) for r in RULES}
        if files['fedavg'] is None:
            continue
        names = files['fedavg']['classes']
        res = out.setdefault(ds, {'scores': {}, 'vs_fedavg': {}, 'forward': {}, 'forgetting': {},
                                  'cost': {}, 'memory': {}, 'adaptations': {}})
        alphas = sorted({k.split('/')[0] for k in files['fedavg']['runs']}, key=float)
        for al in alphas:
            for arm in ARMS:
                per = {}
                for r, f in files.items():
                    if f is None:
                        continue
                    cells = {k.split('/')[1]: c for k, c in f['runs'].items()
                             if k.split('/')[0] == al and arm in c}
                    if cells:
                        per[r] = cells
                if 'fedavg' not in per:
                    continue
                seeds = sorted(per['fedavg'], key=int)
                key = f'{al}/{arm}'
                res['scores'][key], res['vs_fedavg'][key] = {}, {}
                res['forward'][key], res['forgetting'][key] = {}, {}
                res['cost'][key], res['memory'][key], res['adaptations'][key] = {}, {}, {}
                base = np.array([per['fedavg'][s][arm]['final']['macro_f1'] for s in seeds])
                for r, cells in per.items():
                    if sorted(cells, key=int) != seeds:
                        continue                              # incomplete: never mix seed sets
                    v = np.array([cells[s][arm]['final']['macro_f1'] for s in seeds])
                    res['scores'][key][r] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)),
                                             'per_seed': v.tolist(), 'seeds': seeds}
                    if r != 'fedavg':
                        res['vs_fedavg'][key][r] = interval(v - base)
                    runs = [cells[s][arm] for s in seeds]
                    if all(x.get('forward') for x in runs):
                        res['forward'][key][r] = {m: float(np.mean([forward_mean(x, m) for x in runs]))
                                                  for m in ('macro_f1', 'detection', 'false_alarm')}
                    fg = [forgetting(cells[s], arm, names) for s in seeds]
                    res['forgetting'][key][r] = {'mean_forgetting': float(np.nanmean([f['mean_forgetting'] for f in fg])),
                                                 'learned': float(np.mean([f['learned'] for f in fg])),
                                                 'lost': float(np.mean([f['lost'] for f in fg]))}
                    res['cost'][key][r] = float(np.mean([x.get('grad_steps') or 0 for x in runs]))
                    pm = [peak_memory(x) for x in runs]
                    if all(p is not None for p in pm):
                        res['memory'][key][r] = float(np.mean(pm))
                    if all(x.get('detected') is not None for x in runs):
                        res['adaptations'][key][r] = float(np.mean([len(x['detected']) for x in runs]))
    # Holm over the primary family: rehearsal, alpha 0.1 and 100, every dataset and rule
    fam = [out[ds]['vs_fedavg'][f'{al}/rehearsal'][r] for ds in out for al in PRIMARY_ALPHAS
           for r in RULES[1:] if r in out[ds]['vs_fedavg'].get(f'{al}/rehearsal', {})]
    for v, q in zip(fam, holm([v['p'] for v in fam])):
        v['p_holm_primary'] = q
    path = os.path.join(CKPT, a.out)
    json.dump(out, open(path, 'w'), indent=1)
    for ds, res in out.items():
        print(f'\n=== {ds}')
        for key, d in res['vs_fedavg'].items():
            sc = res['scores'][key]
            print(f"  {key:<16} " + '  '.join(f"{r}={v['mean']:.4f}" for r, v in sc.items()))
            print('     ' + ' | '.join(f"{r}-fedavg {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]"
                                       + ('*' if v['clear'] else '') for r, v in d.items()))
    print(f'[saved] {path}  primary family size {len(fam)}')


if __name__ == '__main__':
    main()
