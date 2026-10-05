"""
analyse_final_eval.py -- computes every quantity declared in FINAL_EVAL_PROTOCOL.md
(v1) and FINAL_EVAL_PROTOCOL_v2.md from final_eval.py result files, and writes one
JSON summary.

    python analyse_final_eval.py final_cicids.json final_unsw.json final_ids2018.json
    python analyse_final_eval.py --out final_v2_R2_analysis.json v2/R2_cicids.json ...
    python analyse_final_eval.py --out final_v2_R1_analysis.json --before final_eval_analysis.json v2/R1_cicids.json ...

With --before, each arm's score and each contrast is also given as a paired
change against the earlier analysis (same seeds): the before/after accounting.
"""
import argparse, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, 'checkpoints')
ORDERED = ['nonstat', 'rehearsal', 'cda', 'cda_q1', 'fliids', 'fliids_norm', 'fliids_ce',
           'fliids_ce_half', 'fliids_gb', 'fliids_ce_ls', 'fliids_ce_ls_norm']   # concept histories
CONTRASTS = [('B2-B', 'rehearsal', 'nonstat'), ('C5-B2', 'cda', 'rehearsal'),
             ('C1-B2', 'cda_q1', 'rehearsal'), ('B2-A2', 'rehearsal', 'iid_stream'),
             ('F-B2', 'fliids', 'rehearsal'), ('F-C5', 'fliids', 'cda'),
             ('F-F0', 'fliids', 'fliids_ce'), ('F0-B2', 'fliids_ce', 'rehearsal'),
             # protocol v2 additions
             ('Fn-B2', 'fliids_norm', 'rehearsal'), ('Fn-F0', 'fliids_norm', 'fliids_ce'),
             ('F0h-F0', 'fliids_ce_half', 'fliids_ce'), ('Fgb-F0h', 'fliids_gb', 'fliids_ce_half'),
             ('Fls-F0h', 'fliids_ce_ls', 'fliids_ce_half'),
             ('Flsn-F0h', 'fliids_ce_ls_norm', 'fliids_ce_half'),
             ('F-Fgb', 'fliids', 'fliids_gb'), ('Fn-Fgb', 'fliids_norm', 'fliids_gb')]
FORWARD_CONTRASTS = [('B2-B', 'rehearsal', 'nonstat'), ('C5-B2', 'cda', 'rehearsal'),
                     ('F-B2', 'fliids', 'rehearsal'), ('Fn-B2', 'fliids_norm', 'rehearsal')]
COST_ARMS = ('rehearsal', 'fliids', 'fliids_norm', 'cda', 'nonstat', 'fliids_ce')
PRIMARY = [('B2-B', '0.5', None), ('C5-B2', '0.5', None), ('C5-B2', '0.1', ('cicids', 'unsw')),
           ('F-F0', '0.1', ('cicids', 'unsw')), ('F-F0', '0.5', ('cicids', 'unsw')),
           ('Fn-F0', '0.1', ('cicids', 'unsw')), ('Fn-F0', '0.5', ('cicids', 'unsw'))]


def interval(d):
    """Mean, SD and 95% t-interval of paired differences d (one per seed)."""
    d = np.asarray(d, float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float('nan')
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float('nan')
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if n > 1 and sd > 0 else (1.0 if n > 1 else float('nan'))
    return {'mean': m, 'sd': sd, 'lo': m - h, 'hi': m + h, 'n': n, 'p': p,
            'clear': bool(n > 1 and (m - h > 0 or m + h < 0))}


def summary(v):
    v = np.asarray(v, float)
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0, 'n': len(v)}


def cells_at(d, A):
    c = {k.split('/')[1]: v for k, v in d['runs'].items() if k.split('/')[0] == A}
    return [c[s] for s in sorted(c, key=int)], sorted(c, key=int)


def final_f1(d, alpha, arm):
    """Per-seed final macro-F1, seeds in sorted order (paired across arms)."""
    cells, seeds = cells_at(d, f'{alpha:g}')
    return seeds, np.array([c[arm]['final']['macro_f1'] for c in cells])


def forward_mean(r, key):
    f = [x[key] for x in r.get('forward', []) if x.get(key) is not None]
    return float(np.mean(f)) if f else float('nan')


def peak_memory(r):
    if r.get('memory_stored'):
        return max(max(x) for x in r['memory_stored'])
    rows = [h.get('memory_rows') for h in r.get('history', []) if h.get('memory_rows')]
    return max(max(x) for x in rows) if rows else None


def forgetting(cell, arm, names, key='rec_'):
    """Class-by-class forgetting for one run (protocol v1, quantity 4)."""
    concepts = cell['concepts']
    hist = cell[arm]['history']
    by_event = {r.get('event'): r for r in hist}
    # align one score row to every concept end; a concept with no row (the
    # detector did not fire, so nothing trained) keeps the previous model's scores
    rows, last = [], None
    for i, c in enumerate(concepts):
        r = by_event.get(c['name'])
        if r is None and i == 0:
            r = by_event.get('Monday_init') or by_event.get(concepts[0]['name'] + '_init') or hist[0]
        last = r if r is not None else last
        rows.append(last)
    T = len(concepts)
    intro = {}
    for i, c in enumerate(concepts):
        for k in c['classes']:
            intro.setdefault(k, i)
    per_class = {}
    for k, t0 in intro.items():
        if t0 >= T - 1:
            continue                                   # introduced in the last concept
        nm = names[k]
        s = [rows[t].get(key + nm) for t in range(T)]
        s = [np.nan if v is None else v for v in s]
        after, best, final = s[t0], np.nanmax(s[t0:T - 1]), s[T - 1]
        per_class[nm] = {'introduced': t0, 'right_after': after, 'best': float(best),
                         'final': final, 'forgetting': float(max(0.0, best - final)),
                         'learned_then_lost': bool(best >= 0.5 and final < 0.5)}
    f = [v['forgetting'] for v in per_class.values()]
    return {'mean_forgetting': float(np.mean(f)) if f else float('nan'),
            'lost': int(sum(v['learned_then_lost'] for v in per_class.values())),
            'learned': int(sum(v['best'] >= 0.5 for v in per_class.values())),
            'classes': per_class}


def holm(ps):
    """Holm-adjusted p-values, same order as given."""
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def analyse(fn):
    d = json.load(open(os.path.join(CKPT, fn)))
    names = d['classes']
    alphas = sorted({float(k.split('/')[0]) for k in d['runs']})
    arms = list(dict.fromkeys(a for c in d['runs'].values() for a in c if a != 'concepts'))
    res = {'meta': d['meta'], 'alphas': alphas, 'scores': {}, 'contrasts': {},
           'forgetting': {}, 'cost': {}, 'forward': {}, 'forward_contrasts': {},
           'memory': {}, 'adaptations': {}}
    for al in alphas:
        A = f'{al:g}'
        cells, seeds = cells_at(d, A)
        res['scores'][A] = {}
        for arm in arms:
            if not all(arm in c for c in cells):
                continue
            v = np.array([c[arm]['final']['macro_f1'] for c in cells])
            res['scores'][A][arm] = {**summary(v), 'per_seed': v.tolist(), 'seeds': seeds}
        sc = {a: np.array(x['per_seed']) for a, x in res['scores'][A].items()}
        res['contrasts'][A] = {lab: interval(sc[x] - sc[y]) for lab, x, y in CONTRASTS
                               if x in sc and y in sc and len(sc[x]) == len(sc[y])}
        # forward-stream (runs without forward records, e.g. protocol v1, are skipped)
        fw = {}
        for arm in res['scores'][A]:
            if all(c[arm].get('forward') for c in cells):
                fw[arm] = {k: np.array([forward_mean(c[arm], k) for c in cells])
                           for k in ('macro_f1', 'detection', 'false_alarm')}
        res['forward'][A] = {a: {k: {**summary(v), 'per_seed': v.tolist()} for k, v in m.items()}
                             for a, m in fw.items()}
        res['forward_contrasts'][A] = {f'{lab}/{k}': interval(fw[x][k] - fw[y][k])
                                       for lab, x, y in FORWARD_CONTRASTS if x in fw and y in fw
                                       for k in ('macro_f1', 'detection')}
        res['memory'][A] = {a: summary([peak_memory(c[a]) for c in cells])
                            for a in res['scores'][A]
                            if all(peak_memory(c[a]) is not None for c in cells)}
        res['adaptations'][A] = {a: summary([len(c[a]['detected']) for c in cells])
                                 for a in ('rehearsal', 'cda', 'cda_q1')
                                 if a in res['scores'][A] and all(c[a].get('detected') is not None for c in cells)}
        res['forgetting'][A] = {}
        for arm in ORDERED:
            if arm not in res['scores'][A]:
                continue
            rr = [forgetting(c, arm, names) for c in cells]
            ff = [forgetting(c, arm, names, key='f1_') for c in cells]
            res['forgetting'][A][arm] = {
                'mean_forgetting_recall': summary([r['mean_forgetting'] for r in rr]),
                'mean_forgetting_f1': summary([r['mean_forgetting'] for r in ff]),
                'learned': summary([r['learned'] for r in rr]),
                'learned_then_lost': summary([r['lost'] for r in rr]),
                'per_class_recall': {nm: {q: float(np.nanmean([r['classes'][nm][q] for r in rr
                                                                if nm in r['classes']]))
                                          for q in ('right_after', 'best', 'final', 'forgetting')}
                                     for nm in rr[0]['classes']}}
        res['cost'][A] = {arm: summary([c[arm].get('grad_steps') or 0 for c in cells])
                          for arm in res['scores'][A]}
    # cost of non-IID: score(100) - score(alpha), per method, paired by seed
    if 100.0 in alphas:
        res['noniid_cost'] = {}
        for arm in COST_ARMS:
            if arm not in res['scores'].get('100', {}):
                continue
            s100 = np.array(res['scores']['100'][arm]['per_seed'])
            res['noniid_cost'][arm] = {f'{al:g}': interval(s100 - np.array(res['scores'][f'{al:g}'][arm]['per_seed']))
                                       for al in alphas if al != 100.0 and arm in res['scores'][f'{al:g}']}
        # difference in differences: how much more heterogeneity costs FL-IIDS than rehearsal
        res['noniid_cost_vs_B2'] = {}
        b2 = np.array(res['scores']['100'].get('rehearsal', {}).get('per_seed', []))
        for arm in ('fliids', 'fliids_norm'):
            if arm not in res['scores']['100'] or not len(b2):
                continue
            f100 = np.array(res['scores']['100'][arm]['per_seed'])
            res['noniid_cost_vs_B2'][arm] = {
                f'{al:g}': interval((f100 - np.array(res['scores'][f'{al:g}'][arm]['per_seed']))
                                    - (b2 - np.array(res['scores'][f'{al:g}']['rehearsal']['per_seed'])))
                for al in alphas if al != 100.0 and arm in res['scores'][f'{al:g}']}
    return d['meta']['dataset'], res


def before_after(now, before):
    """Paired change of every arm score and contrast against an earlier analysis."""
    out = {}
    for A, arms in now['scores'].items():
        prev = before.get('scores', {}).get(A, {})
        for arm, v in arms.items():
            if arm in prev and prev[arm]['seeds'] == v['seeds']:
                out.setdefault(A, {})[arm] = interval(np.array(v['per_seed']) - np.array(prev[arm]['per_seed']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--out', default='final_eval_analysis.json')
    ap.add_argument('--before', default=None, help='earlier analysis JSON (same seeds)')
    a = ap.parse_args()
    out = {}
    for fn in a.files:
        ds, res = analyse(fn)
        out[ds] = res
    # Holm-adjusted p-values over the primary family (protocol v2, sensitivity only)
    fam = [out[ds]['contrasts'][A][lab] for lab, A, only in PRIMARY for ds in out
           if (only is None or ds in only) and lab in out[ds]['contrasts'].get(A, {})]
    # P4: how much more heterogeneity costs FL-IIDS than rehearsal, alpha 0.1
    fam += [out[ds]['noniid_cost_vs_B2'][arm]['0.1'] for ds in ('cicids', 'unsw') if ds in out
            for arm in ('fliids', 'fliids_norm') if '0.1' in out[ds].get('noniid_cost_vs_B2', {}).get(arm, {})]
    if fam:
        for v, q in zip(fam, holm([v['p'] for v in fam])):
            v['p_holm_primary'] = q
    if a.before:
        prev = json.load(open(os.path.join(CKPT, a.before)))
        for ds in out:
            if ds in prev:
                out[ds]['before_after'] = before_after(out[ds], prev[ds])
    path = os.path.join(CKPT, a.out)
    json.dump(out, open(path, 'w'), indent=1)
    for ds, res in out.items():
        print(f'\n=== {ds}')
        for A in res['scores']:
            print(f'  alpha={A}: ' + '  '.join(f"{k}={v['mean']:.4f}±{v['sd']:.4f}"
                                              for k, v in res['scores'][A].items()))
            for k, v in res['contrasts'][A].items():
                print(f"     {k:<8} {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]"
                      + ('  clear' if v['clear'] else ''))
            for arm, f in res['forgetting'][A].items():
                print(f"     forget {arm:<17} recall {f['mean_forgetting_recall']['mean']:.3f} "
                      f"lost {f['learned_then_lost']['mean']:.1f} of {f['learned']['mean']:.1f} learned")
    print(f'[saved] {path}')


if __name__ == '__main__':
    main()
