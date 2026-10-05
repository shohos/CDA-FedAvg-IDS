"""
compare_crash_cell.py -- after a crash and resume, check that the affected cell,
rerun from scratch, is bit-identical to the saved results (timing fields ignored).

    python compare_crash_cell.py agg/fedavg_ids2018.json agg_crashcheck/fedavg_ids2018_s805.json 100/805
"""
import json, os, sys

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
IGNORE = {'seconds', 'started', 'finished', 'wall', 'time'}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in IGNORE}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def diff(a, b, path=''):
    if type(a) != type(b):
        return [f'{path}: type {type(a).__name__} vs {type(b).__name__}']
    if isinstance(a, dict):
        out = [f'{path}/{k}: missing on one side' for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += diff(a[k], b[k], f'{path}/{k}')
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} vs {len(b)}']
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f'{path}[{i}]')]
    return [] if a == b else [f'{path}: {a!r} vs {b!r}']


if __name__ == '__main__':
    saved, rerun, key = sys.argv[1:4]
    A = json.load(open(os.path.join(CKPT, saved)))['runs'][key]
    B = json.load(open(os.path.join(CKPT, rerun)))['runs'][key]
    arms = [k for k in A if k != 'concepts']
    ok = True
    for arm in arms + ['concepts']:
        d = diff(strip(A[arm]), strip(B.get(arm)))
        print(f'{arm:<10} {"IDENTICAL" if not d else "DIFFERS (" + str(len(d)) + ")"}')
        for line in d[:5]:
            print('   ', line)
        ok &= not d
    print('ALL IDENTICAL' if ok else 'NOT IDENTICAL')
    sys.exit(0 if ok else 1)
