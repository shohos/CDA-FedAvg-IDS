"""
ids2018_loader.py — CSE-CIC-IDS2018 as a 7-concept drift stream.

Generalisation test for the whole pipeline: a different capture, a different
attack schedule, and a longer drift sequence than CICIDS2017's four transitions.

Design decisions, each forced by how 2018 actually differs from 2017 (verified
from the per-day label counts, not assumed):

  * No PortScan; Infiltration is a MAJOR class (161,934 flows) where in 2017 it
    had 36 and was dropped as unlearnable. So the class set is not 2017's.
  * No benign-only day — 2017's Monday baseline has no equivalent. Concept 0 is
    therefore built from 14-02's benign traffic, with that day's attacks forming
    concept 1. Every later concept carries its own day's benign traffic, exactly
    as BENIGN appears in every 2017 concept.
  * Rows whose Label is literally "Label" are repeated CSV headers — dropped.
  * 80 columns; dropping Timestamp and Label leaves 78 features, the same count
    as 2017, so IDSDNN and the pipeline are reused unchanged. The features are
    NOT identical in meaning across the two captures, so weights cannot transfer
    between datasets — only the architecture and the protocol do.

7 concepts x 5 rounds = 35 rounds, preserving R_PER_CONCEPT = 5.

The 20-02 file is 4 GB, so files are read in chunks and subsampled per class as
they stream rather than loaded whole.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_2018 = os.path.join(ROOT, "CICIDS2018")
SEED = 42

CLASS_NAMES_2018 = ['Benign', 'BruteForce', 'DoS', 'DDoS',
                    'WebAttack', 'Infiltration', 'Bot']
CLASS_TO_IDX_2018 = {c: i for i, c in enumerate(CLASS_NAMES_2018)}
N_CLASSES_2018 = len(CLASS_NAMES_2018)
N_FEATURES_2018 = 78

# The 20-02 capture ships 84 columns: it alone carries these identifiers. Every
# other day has 80. Left in place they become "features", to_numeric turns IPs
# into NaN, the finite-check drops every row, and the day silently vanishes.
IDENT_COLS_2018 = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp']

_F = 'TrafficForML_CICFlowMeter.csv'
CONCEPTS_2018 = [
    ('Baseline',     [f'Wednesday-14-02-2018_{_F}'], 'Benign baseline', True),
    ('BruteForce',   [f'Wednesday-14-02-2018_{_F}'], 'FTP/SSH brute force', False),
    ('DoS',          [f'Thursday-15-02-2018_{_F}',
                      f'Friday-16-02-2018_{_F}'],    'DoS GoldenEye/Slowloris/Hulk', False),
    ('DDoS',         [f'Thuesday-20-02-2018_{_F}',
                      f'Wednesday-21-02-2018_{_F}'], 'DDoS LOIC/HOIC', False),
    ('WebAttack',    [f'Thursday-22-02-2018_{_F}',
                      f'Friday-23-02-2018_{_F}'],    'Web brute force / XSS / SQLi', False),
    ('Infiltration', [f'Wednesday-28-02-2018_{_F}',
                      f'Thursday-01-03-2018_{_F}'],  'Infiltration', False),
    ('Bot',          [f'Friday-02-03-2018_{_F}'],    'Bot', False),
]
N_CONCEPTS_2018 = len(CONCEPTS_2018)

# One concept per capture day — the design CICIDS2017 uses, applied to all 10
# days. Day 1 already contains attacks (2018 has no benign-only day), so concept 0
# is simply the first day rather than a benign baseline.
#
# Valuable property: consecutive days often repeat the previous day's attack
# class, so the stream contains BOTH real drifts and non-drifts:
#   new class introduced -> days 2 (DoS), 4 (DDoS), 6 (WebAttack),
#                           8 (Infiltration), 10 (Bot)   = 5 true drifts
#   no new class         -> days 3, 5, 7, 9              = 4 true non-drifts
# That makes detection measurable as TP and FP on real data, not a synthetic
# null stream.
CONCEPTS_2018_DAILY = [
    ('D1-14Feb-BruteForce',   [f'Wednesday-14-02-2018_{_F}'], 'FTP/SSH brute force',   False),
    ('D2-15Feb-DoS',          [f'Thursday-15-02-2018_{_F}'],  'DoS GoldenEye/Slowloris', False),
    ('D3-16Feb-DoS',          [f'Friday-16-02-2018_{_F}'],    'DoS Hulk/SlowHTTPTest',   False),
    ('D4-20Feb-DDoS',         [f'Thuesday-20-02-2018_{_F}'],  'DDoS LOIC-HTTP',          False),
    ('D5-21Feb-DDoS',         [f'Wednesday-21-02-2018_{_F}'], 'DDoS HOIC/LOIC-UDP',      False),
    ('D6-22Feb-WebAttack',    [f'Thursday-22-02-2018_{_F}'],  'Web brute force/XSS/SQLi', False),
    ('D7-23Feb-WebAttack',    [f'Friday-23-02-2018_{_F}'],    'Web brute force/XSS/SQLi', False),
    ('D8-28Feb-Infiltration', [f'Wednesday-28-02-2018_{_F}'], 'Infiltration',            False),
    ('D9-01Mar-Infiltration', [f'Thursday-01-03-2018_{_F}'],  'Infiltration',            False),
    ('D10-02Mar-Bot',         [f'Friday-02-03-2018_{_F}'],    'Bot',                     False),
]
N_CONCEPTS_2018_DAILY = len(CONCEPTS_2018_DAILY)

# transitions INTO these concept indices introduce a class not seen before
TRUE_DRIFT_IDX = {1, 3, 5, 7, 9}          # days 2, 4, 6, 8, 10
TRUE_NONDRIFT_IDX = {2, 4, 6, 8}          # days 3, 5, 7, 9


def normalize_label_2018(raw):
    l = str(raw).strip()
    if l in ('Label', '', 'nan'):
        return None                       # repeated CSV header rows
    if l == 'Benign':                     return 'Benign'
    if 'BruteForce' in l or 'Bruteforce' in l:
        return 'WebAttack' if 'Web' in l else 'BruteForce'
    if 'Brute Force' in l:                return 'WebAttack'
    if 'SQL' in l or 'XSS' in l:          return 'WebAttack'
    if 'DDOS' in l.upper():               return 'DDoS'
    if 'DoS' in l:                        return 'DoS'
    if 'Infilteration' in l or 'Infiltration' in l:
        return 'Infiltration'
    if l == 'Bot':                        return 'Bot'
    return None


def _read_day(path, max_per_class, benign_only, rng, chunk=500_000):
    """Stream one day's CSV, keeping at most max_per_class rows per class."""
    keep_X = {c: [] for c in range(N_CLASSES_2018)}
    keep_n = {c: 0 for c in range(N_CLASSES_2018)}
    feat_cols = None
    for ch in pd.read_csv(path, chunksize=chunk, low_memory=False):
        ch.columns = [c.strip() for c in ch.columns]
        ch = ch.drop(columns=[c for c in IDENT_COLS_2018 if c in ch.columns])
        cls = ch['Label'].map(normalize_label_2018)
        ch = ch[cls.notna()]
        cls = cls[cls.notna()]
        if feat_cols is None:
            feat_cols = [c for c in ch.columns if c != 'Label']
        Xc = ch[feat_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
        yc = np.array([CLASS_TO_IDX_2018[c] for c in cls], dtype=np.int64)
        ok = np.isfinite(Xc).all(axis=1)
        Xc, yc = Xc[ok], yc[ok]
        if benign_only:
            sel = yc == CLASS_TO_IDX_2018['Benign']
            Xc, yc = Xc[sel], yc[sel]
        for c in np.unique(yc):
            if keep_n[c] >= max_per_class:
                continue
            idx = np.where(yc == c)[0]
            take = min(len(idx), max_per_class - keep_n[c])
            sub = rng.choice(idx, take, replace=False) if take < len(idx) else idx
            keep_X[c].append(Xc[sub])
            keep_n[c] += take
        if all(keep_n[c] >= max_per_class for c in keep_n if keep_n[c] > 0) and \
           sum(keep_n.values()) >= max_per_class * 2:
            break                          # enough of everything seen so far
    Xs = [np.concatenate(v) for v in keep_X.values() if v]
    ys = [np.full(len(np.concatenate(v)), c, dtype=np.int64)
          for c, v in keep_X.items() if v]
    if not Xs:
        return np.zeros((0, N_FEATURES_2018), np.float32), np.zeros(0, np.int64), feat_cols
    return np.concatenate(Xs), np.concatenate(ys), feat_cols


def load_ids2018_dataset(n_clients=10, alpha=0.5, max_per_class=5000,
                         test_fraction=0.2, data_dir=DATA_DIR_2018, verbose=True,
                         mode='daily'):
    """Same return shape as load_ids_dataset: (concepts, test_set, {}, scaler, feat_cols)."""
    from ids_loader import dirichlet_partition
    rng = np.random.default_rng(SEED)
    spec = CONCEPTS_2018_DAILY if mode == 'daily' else CONCEPTS_2018
    raw = []
    cols = None
    for name, files, desc, benign_only in spec:
        Xs, ys = [], []
        for f in files:
            X, y, feat_cols = _read_day(os.path.join(data_dir, f),
                                        max_per_class, benign_only, rng)
            # _read_day orders features by each file's OWN header and the rows
            # are then concatenated, so every file must agree on the order or
            # its rows land with features in the wrong columns. All ten day
            # files do agree today (checked 2026-09-17); this makes a future
            # file that does not fail loudly instead of misaligning silently.
            if feat_cols is not None:
                if cols is None:
                    cols = list(feat_cols)
                elif list(feat_cols) != cols:
                    raise RuntimeError(
                        f"{f}: feature order differs from earlier 2018 files; "
                        f"concatenating would misalign columns")
            if len(X):
                Xs.append(X); ys.append(y)
        if not Xs:
            raise RuntimeError(
                f"concept '{name}' loaded ZERO rows from {files}. A silently "
                f"skipped concept shifts every downstream index — check the "
                f"column schema of that file.")
        X, y = np.concatenate(Xs), np.concatenate(ys)
        if not benign_only:
            # cap benign so it does not swamp the concept, as 2017 does
            keep = []
            for c in range(N_CLASSES_2018):
                idx = np.where(y == c)[0]
                if len(idx) == 0:
                    continue
                if len(idx) > max_per_class:
                    idx = rng.choice(idx, max_per_class, replace=False)
                keep.append(idx)
            k = np.sort(np.concatenate(keep))
            X, y = X[k], y[k]
        if verbose:
            u, cnt = np.unique(y, return_counts=True)
            print(f"  {name:<14}{len(X):>8,} | "
                  f"{ {CLASS_NAMES_2018[a]: int(b) for a, b in zip(u, cnt)} }")
        raw.append((name, X, y, desc))

    test_X, test_y, train_parts = [], [], []
    for name, X, y, desc in raw:
        if len(np.unique(y)) >= 2 and len(X) >= 20:
            try:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_fraction, stratify=y, random_state=SEED)
            except ValueError:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_fraction, random_state=SEED)
        else:
            n_te = max(2, int(len(X) * test_fraction))
            s = rng.choice(len(X), n_te, replace=False)
            t = np.setdiff1d(np.arange(len(X)), s)
            Xtr, Xte, ytr, yte = X[t], X[s], y[t], y[s]
        test_X.append(Xte); test_y.append(yte)
        train_parts.append((name, Xtr, ytr, desc))

    # rows with any non-finite value were already dropped one by one in _read_day,
    # which uses no other rows. The scaler follows ids_loader.PREPROC (audit F1).
    import ids_loader
    scaler = StandardScaler()
    scaler.fit(train_parts[0][1] if ids_loader.PREPROC == 'causal'
               else np.concatenate([x for _, x, _, _ in train_parts]))
    X_test = scaler.transform(np.concatenate(test_X))
    y_test = np.concatenate(test_y)

    concepts = []
    for (name, Xtr, ytr, desc), Xte, yte in zip(train_parts, test_X, test_y):
        concepts.append({'name': name, 'description': desc,
                         'client_data': dirichlet_partition(
                             scaler.transform(Xtr), ytr, n_clients, alpha, rng),
                         'test': (scaler.transform(Xte), yte)})
    if verbose:
        print(f"\nReady: {len(concepts)} concepts x {n_clients} clients "
              f"(Dirichlet α={alpha}) | test {len(X_test):,}")
    return concepts, (X_test, y_test), {}, scaler, cols
