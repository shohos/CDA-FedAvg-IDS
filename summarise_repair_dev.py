"""
summarise_repair_dev.py -- tables from the audit-repair development runs
(checkpoints/audit_repair_dev/), consumed seeds only. Prints; writes summary.json.

  repaired_<ds>.json  every arm under --impl repaired: retention score (final
                      pooled macro-F1), forward-stream scores, stored memory
  decomp_<ds>.json    which repair moved which number (dev_repairs.py)
"""
import json, os
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'audit_repair_dev')


def ms(v):
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) if v else None


def peak_memory(r):
    if 'memory_stored' in r:                                   # FL-IIDS: rows kept
        return max(max(x) for x in r['memory_stored'])
    rows = [h.get('memory_rows') for h in r.get('history', [])] or r.get('memory_rows', [])
    rows = [x for x in rows if x]
    return max(max(x) for x in rows) if rows else None


def forward(r, key):
    f = [x[key] for x in r.get('forward', []) if x.get(key) is not None]
    return float(np.mean(f)) if f else None


def repaired(ds):
    runs = {}
    for f in (f'repaired_{ds}.json', f'repaired_{ds}_norm.json'):     # _norm: extra arms
        p = os.path.join(D, f)
        if os.path.exists(p):
            for key, cell in json.load(open(p))['runs'].items():
                runs.setdefault(key, {}).update(cell)
    out = {}
    for key, cell in runs.items():
        alpha = key.split('/')[0]
        for arm, r in cell.items():
            if arm == 'concepts':
                continue
            a = out.setdefault(alpha, {}).setdefault(arm, {k: [] for k in
                ('final', 'fwd_f1', 'fwd_det', 'fwd_fa', 'mem', 'steps', 'drifts')})
            a['final'].append(r['final']['macro_f1'])
            a['fwd_f1'].append(forward(r, 'macro_f1'))
            a['fwd_det'].append(forward(r, 'detection'))
            a['fwd_fa'].append(forward(r, 'false_alarm'))
            a['mem'].append(peak_memory(r))
            a['steps'].append(r.get('grad_steps'))
            a['drifts'].append(len(r['detected']) if r.get('detected') is not None else None)
    return {al: {arm: {k: ms(v) for k, v in d.items()} for arm, d in arms.items()}
            for al, arms in out.items()}


def decomp(ds):
    p = os.path.join(D, f'decomp_{ds}.json')
    if not os.path.exists(p):
        return {}
    runs = json.load(open(p))['runs']
    by = {}
    for key, cell in runs.items():
        cfg, seed = key.split('/')
        for arm, r in cell.items():
            by.setdefault(arm, {}).setdefault(cfg, {})[seed] = r['final']['macro_f1']
    out = {}
    for arm, cfgs in by.items():
        base = cfgs.get('legacy', {})
        for cfg, vals in cfgs.items():
            diffs = [vals[s] - base[s] for s in vals if s in base]
            out.setdefault(arm, {})[cfg] = {'mean': ms(list(vals.values())),
                                            'minus_legacy': ms(diffs) if cfg != 'legacy' else None}
    return out


def fmt(t):
    return '   -   ' if t is None else f'{t[0]:.4f}+-{t[1]:.4f}'


if __name__ == '__main__':
    summary = {}
    for ds in ('cicids', 'unsw', 'ids2018'):
        rep, dec = repaired(ds), decomp(ds)
        summary[ds] = {'repaired': rep, 'decomp': dec}
        for al, arms in sorted(rep.items()):
            print(f'\n== {ds} repaired alpha={al}  (retention | forward F1, detection, false alarm | peak rows | steps | drifts)')
            for arm, v in arms.items():
                d = v['drifts']
                print(f"  {arm:<15} {fmt(v['final'])} | {fmt(v['fwd_f1'])} {fmt(v['fwd_det'])} {fmt(v['fwd_fa'])}"
                      f" | {v['mem'][0] if v['mem'] else '-':>7} | {v['steps'][0] if v['steps'] else '-':>8.0f}"
                      f" | {d[0] if d else '-'}  (n={v['final'][2]})")
        for arm, cfgs in dec.items():
            print(f'\n== {ds} decomposition, {arm}: config mean  (change from legacy)')
            for cfg, v in cfgs.items():
                print(f"  {cfg:<8} {fmt(v['mean'])}  {fmt(v['minus_legacy'])}")
    json.dump(summary, open(os.path.join(D, 'summary.json'), 'w'), indent=1)
