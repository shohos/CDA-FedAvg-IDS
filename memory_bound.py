"""
memory_bound.py -- keep a rehearsal memory within L rows (audit F5, 2026-09-26).

The original rule capped each class PER CONCEPT and never evicted, so the memory
grew with the stream: L = 1400 was a per-concept budget, not a bound. evict_to_cap
trims the memory after each new concept to at most `cap` rows. Each class gets an
equal share; a class with fewer rows keeps all of them and its unused share passes
to the others. Rows of an over-quota class are chosen uniformly at random across
the concepts they came from, so old concepts are not always the first to go.

Shared by ids_cda_pipeline and unsw_cda_pipeline so the two cannot drift apart.
"""
import numpy as np


def fair_quota(counts, cap):
    """{class: available} -> {class: kept}. Equal shares, capped by supply."""
    q, left = {}, cap
    order = sorted(counts, key=counts.get)
    for i, c in enumerate(order):
        q[c] = min(counts[c], left // (len(order) - i))
        left -= q[c]
    return q


def evict_to_cap(concepts, cap, rng):
    """concepts: list of (X, y, name). Returns the same structure, at most cap rows."""
    if sum(len(c[1]) for c in concepts) <= cap:
        return concepts
    y = np.concatenate([c[1] for c in concepts])
    src = np.concatenate([np.full(len(c[1]), i) for i, c in enumerate(concepts)])
    keep = np.zeros(len(y), bool)
    for c, n in fair_quota({int(c): int((y == c).sum()) for c in np.unique(y)}, cap).items():
        idx = np.where(y == c)[0]
        keep[idx if n >= len(idx) else rng.choice(idx, n, replace=False)] = True
    out = []
    for i, (X, yc, name) in enumerate(concepts):
        m = keep[src == i]
        if m.any():
            out.append((X[m], yc[m], name))
    return out


if __name__ == '__main__':
    # the audit's counterexample: twenty benign-only concepts of 100 rows each
    rng = np.random.default_rng(0)
    mem = []
    for t in range(20):
        mem = evict_to_cap(mem + [(np.full((100, 3), t), np.zeros(100, int), f'c{t}')], 1400, rng)
        assert sum(len(c[1]) for c in mem) <= 1400
    assert sum(len(c[1]) for c in mem) == 1400
    assert mem[0][0][0, 0] == 0 and len(mem) == 20       # old concepts still represented
    # a rare class keeps all it has; the others split what is left
    assert fair_quota({0: 5000, 1: 30, 2: 900}, 1400) == {1: 30, 2: 685, 0: 685}
    print('memory_bound ok')
