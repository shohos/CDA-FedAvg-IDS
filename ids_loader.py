"""
IDS Data Loader for CDA-FedAvg extension to CICIDS2017.

Inspection facts (from ids_inspect.py output):
  - Label column: ' Label'  (leading space — stripped here)
  - 78 numeric features + 1 label = 79 columns total
  - Confirmed inf values in Flow Bytes/s, Flow Packets/s
  - Web Attack labels contain Windows encoding artifact — matched by content
  - Infiltration (36 samples) and Heartbleed (11) dropped as unlearnable
  - 7 meaningful classes after grouping → PER_CLASS = L/(2M) = 100 (same as HAR)

Concept drift mapping (5 days = 5 HAR positions):
  Concept 0: Monday    → BENIGN baseline (no attacks)
  Concept 1: Tuesday   → BruteForce (FTP-Patator, SSH-Patator)
  Concept 2: Wednesday → DoS variants (Hulk, GoldenEye, slowloris, Slowhttptest)
  Concept 3: Thursday  → WebAttack (Brute Force, XSS, Sql Injection)
  Concept 4: Friday    → DDoS + PortScan + Bot
"""

import os, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
                             recall_score)

# ── Paths ─────────────────────────────────────────────────────────────────────

# ROOT = this file's directory, so the same code runs on Colab (Drive) and locally.
ROOT     = os.path.dirname(os.path.abspath(__file__))
# ── Dataset version ───────────────────────────────────────────────────────────
# 'original' = the CICIDS2017 CSVs as published by CIC.
# 'improved' = Engelen et al., WTMC 2021 — corrected flow construction and
#   labelling; over 20% of records revised. Same 5-day structure and the same 78
#   features (renamed by a newer CICFlowMeter), so the concept design carries over
#   unchanged. Verified: day -> attack mapping is identical.
DATA_VERSION = 'improved'      # 'improved' = Engelen et al. corrected labels,
                               # now the default: on corrected data every arm
                               # gains ~0.07-0.10 macro-F1 and seed variance
                               # collapses (cda 0.8750±0.0502 -> 0.9742±0.0045).
                               # 'original' reproduces the published runs.

# Engelen adds a ' - Attempted' suffix for attacks launched but unsuccessful.
# 'merge' folds them into the base class (closest to the original labelling),
# 'drop' discards them, 'separate' keeps the suffix and they fall through to None.
ATTEMPTED = 'merge'            # 'merge' | 'drop' | 'separate'

DATA_DIR = os.path.join(ROOT, "MachineLearningCVE")
DATA_DIR_IMPROVED = os.path.join(ROOT, "CICIDS2017-improved")
IDENT_COLS = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Src Port']

CONCEPT_FILES_IMPROVED = [
    ('Monday',    ['Monday-WorkingHours.csv'],    'BENIGN baseline (no attacks yet)'),
    ('Tuesday',   ['Tuesday-WorkingHours.csv'],   'BruteForce attacks emerge (FTP/SSH-Patator)'),
    ('Wednesday', ['Wednesday-WorkingHours.csv'], 'DoS attacks emerge'),
    ('Thursday',  ['Thursday-WorkingHours.csv'],  'Web Attacks emerge'),
    ('Friday',    ['Friday-WorkingHours.csv'],    'DDoS, PortScan, Bot attacks emerge'),
]


# ── Concept granularity ───────────────────────────────────────────────────────
# '5day'     one concept per capture day (as published).
# '8session' one concept per capture SESSION. The original CICIDS2017 ships 8
#            CSVs because Thursday and Friday were captured in separate
#            morning/afternoon sessions; the corrected release merges them per
#            day, but the Timestamp column recovers the split.
#
# Why 8 is the better design:
#   * Friday holds THREE distinct attacks in three sessions — Bot, PortScan,
#     DDoS — which '5day' collapses into a single drift event, discarding two
#     detection events and the "model has seen attacks, but not this one" case.
#   * Thursday afternoon contains 31 Infiltration flows, which we drop as
#     unlearnable, leaving it effectively BENIGN-ONLY. That is a genuine
#     NON-DRIFT transition — and '5day' has none at all, so detection rate is
#     the only measurable quantity, which is exactly the metric an always-firing
#     detector maximises. 8 sessions give 6 drifts and 1 true negative.
CONCEPT_MODE = '5day'          # '5day' | '8session'

# Preprocessing policy (audit F1, 2026-09-26).
# 'all_train' reproduces the original runs: the scaler saw the training rows of
#   every concept before the stream began, and infinite values were replaced by
#   each concept's column maximum computed before the train/test split.
# 'causal': the fill values and the scaler come from the FIRST concept's training
#   rows only and are then frozen, so no later or held-out data shapes the inputs.
#   Rows selected, split and partitioned are identical under both policies.
PREPROC = 'causal'             # 'causal' | 'all_train'

# (name, files, hour_range or None, description)
CONCEPT_FILES_8SESSION = [
    ('Monday',        ['Monday-WorkingHours.csv'],    None,     'BENIGN baseline'),
    ('Tuesday',       ['Tuesday-WorkingHours.csv'],   None,     'BruteForce emerges'),
    ('Wednesday',     ['Wednesday-WorkingHours.csv'], None,     'DoS emerges'),
    ('Thu-WebAttack', ['Thursday-WorkingHours.csv'],  (0, 17),  'Web attacks (morning)'),
    ('Thu-Benign',    ['Thursday-WorkingHours.csv'],  (18, 23), 'Infiltration dropped -> benign only (NON-DRIFT)'),
    ('Fri-Bot',       ['Friday-WorkingHours.csv'],    (0, 17),  'Bot emerges'),
    ('Fri-PortScan',  ['Friday-WorkingHours.csv'],    (18, 19), 'PortScan emerges'),
    ('Fri-DDoS',      ['Friday-WorkingHours.csv'],    (20, 23), 'DDoS emerges'),
]
# transitions INTO these indices introduce a class not seen before
DRIFT_IDX_8 = {1, 2, 3, 5, 6, 7}
NONDRIFT_IDX_8 = {4}


def _canon(c):
    return c.strip().lower().replace(' ', '').replace('_', '').replace('/', '')

CKPT_DIR = os.path.join(ROOT, "checkpoints")
SEED     = 42

# ── Classes ───────────────────────────────────────────────────────────────────

# ── Normalisation layer ───────────────────────────────────────────────────────
# 'batch' reproduces every published result. 'layer' replaces BatchNorm1d with
# LayerNorm, which carries no running statistics — so FedAvg has nothing
# distribution-dependent to average across heterogeneous clients. Motivation:
# arXiv:2301.02982 (BN damages FL on non-IID data) and FedBN (ICLR 2021).
# Trainable parameter count is identical either way (2*dim per layer), so the
# comparison is parameter-matched.
NORM = 'layer'   # 'batch' | 'layer'  — 'layer' is the best measured
                 # (lr_sweep/opt_sweep); 'batch' reproduces the published runs


def make_norm(dim, norm=None):
    norm = norm or NORM
    return nn.LayerNorm(dim) if norm == 'layer' else nn.BatchNorm1d(dim)


CLASS_NAMES = ['BENIGN', 'DoS', 'DDoS', 'PortScan', 'BruteForce', 'WebAttack', 'Bot']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
N_CLASSES    = len(CLASS_NAMES)    # 7
N_FEATURES   = 78

# ── Concept ordering (Monday → Friday, temporal concept drift) ────────────────

CONCEPT_FILES = [
    (
        'Monday',
        ['Monday-WorkingHours.pcap_ISCX.csv'],
        'BENIGN baseline (no attacks yet)'
    ),
    (
        'Tuesday',
        ['Tuesday-WorkingHours.pcap_ISCX.csv'],
        'BruteForce attacks emerge (FTP/SSH-Patator)'
    ),
    (
        'Wednesday',
        ['Wednesday-workingHours.pcap_ISCX.csv'],
        'DoS attacks emerge (Hulk, GoldenEye, slowloris, Slowhttptest)'
    ),
    (
        'Thursday',
        [
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        ],
        'Web Attacks emerge (Brute Force, XSS, Sql Injection)'
        # Note: Infiltration (36 samples) is dropped in normalize_label
    ),
    (
        'Friday',
        [
            'Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        ],
        'DDoS, PortScan, Bot attacks emerge'
    ),
]

N_CONCEPTS = len(CONCEPT_FILES)    # 5


# ── Label normalisation ───────────────────────────────────────────────────────

def normalize_label(raw: str):
    """
    Map raw CICIDS2017 label string to canonical class name.
    Returns None for classes to be dropped (Infiltration, Heartbleed).

    Robust to the Windows-1252 encoding artifacts that appear as '?'
    in inspection output (e.g., 'Web Attack ? Brute Force').
    Uses content-based matching rather than exact string equality.
    """
    l = str(raw).strip()
    if ' - Attempted' in l or l.endswith('- Attempted'):
        if ATTEMPTED == 'drop':
            return None
        if ATTEMPTED == 'merge':
            l = l.replace(' - Attempted', '').strip()
    if l == 'BENIGN':           return 'BENIGN'
    if l == 'DDoS':             return 'DDoS'  # must check BEFORE DoS substring match
    if 'DoS' in l:              return 'DoS'
    if 'Heartbleed' in l:       return None    # 11 samples — unlearnable
    if l == 'PortScan':         return 'PortScan'
    if 'Patator' in l:          return 'BruteForce'
    if 'Web Attack' in l:       return 'WebAttack'
    if l == 'Bot':              return 'Bot'
    if l == 'Infiltration':     return None    # 36 samples — unlearnable
    return None    # any other unknown label → dropped


# ── CSV loading ───────────────────────────────────────────────────────────────

def finite_max(X):
    """Per-column maximum over finite values (0 for a column with none)."""
    Xm = np.where(np.isfinite(X), X, -np.inf)
    m = Xm.max(axis=0) if len(X) else np.zeros(X.shape[1], X.dtype)
    return np.where(np.isfinite(m), m, 0.0).astype(X.dtype)


def fill_nonfinite(X, fill):
    """Replace inf/NaN entries column-wise with `fill`."""
    return np.where(np.isfinite(X), X, fill[None, :]).astype(X.dtype)


_CANON_MAP = {}


def _build_canon_map():
    """canonical(name) -> original CICIDS2017 column name, built from the header."""
    global _CANON_MAP
    if _CANON_MAP:
        return
    hdr = pd.read_csv(os.path.join(DATA_DIR, CONCEPT_FILES[0][1][0]),
                      nrows=0, encoding='latin-1')
    _CANON_MAP = {_canon(c): c.strip() for c in hdr.columns}


def load_concept_csv(files, data_dir=DATA_DIR, max_per_class=5000, hours=None):
    """
    Load one concept's CSV file(s), clean, and subsample.

    Handles:
      - Leading/trailing whitespace in ALL column names
      - Infinity values in numeric columns (replace with finite column max)
      - Label encoding artifacts (via normalize_label)
      - Per-class subsampling to max_per_class (keeps dataset manageable)

    Returns:
        X: float32 ndarray (N, 78)
        y: int64  ndarray (N,)  — class indices per CLASS_NAMES
        feat_cols: list of 78 feature column names (stripped)
    """
    if DATA_VERSION == 'improved':
        _build_canon_map()
    frames = []
    for fname in files:
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path, encoding='latin-1', low_memory=False)
        df.columns = [c.strip() for c in df.columns]    # ← fix whitespace
        if DATA_VERSION == 'improved':
            if hours is not None and 'Timestamp' in df.columns:
                # '07/07/2017 01:59:50 PM' -> 24h hour, without dateutil fallback
                ts = df['Timestamp'].astype(str)
                hh = ts.str.slice(11, 13).astype(int) % 12
                hh = hh + 12 * ts.str.slice(-2).eq('PM')
                df = df[(hh >= hours[0]) & (hh <= hours[1])]
            df = df.drop(columns=[c for c in IDENT_COLS if c in df.columns])
            # rename the newer CICFlowMeter names back to the canonical ones so
            # both versions expose an identical 78-feature space
            df = df.rename(columns={c: _CANON_MAP[_canon(c)]
                                    for c in df.columns
                                    if _canon(c) in _CANON_MAP})
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Normalise label column (strip + map)
    df['Label'] = df['Label'].astype(str).str.strip()
    df['_cls']  = df['Label'].apply(normalize_label)
    df = df[df['_cls'].notna()].reset_index(drop=True)

    # Feature matrix: all numeric columns except Label/_cls
    feat_cols = [
        c for c in df.columns
        if c not in ('Label', '_cls') and pd.api.types.is_numeric_dtype(df[c])
    ]
    assert len(feat_cols) == N_FEATURES, \
        f"Expected {N_FEATURES} numeric features, got {len(feat_cols)}"

    X = df[feat_cols].values.astype(np.float32)
    y_str = df['_cls'].values

    # Replace inf values with per-column finite maximum ('causal' defers this to
    # after the split, with values from the first concept's training rows)
    if PREPROC == 'all_train':
        X = fill_nonfinite(X, finite_max(X))

    # Encode class labels
    y = np.array([CLASS_TO_IDX[c] for c in y_str], dtype=np.int64)

    # Per-class subsampling
    if max_per_class is not None:
        rng = np.random.default_rng(SEED)
        keep = []
        for cls_idx in range(N_CLASSES):
            idx = np.where(y == cls_idx)[0]
            if len(idx) == 0:
                continue
            if len(idx) > max_per_class:
                idx = rng.choice(idx, max_per_class, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
        X, y = X[keep], y[keep]

    return X, y, feat_cols


# ── Dirichlet non-IID partitioning ────────────────────────────────────────────

def dirichlet_partition(X, y, n_clients, alpha, rng):
    """
    Partition (X, y) into n_clients via Dirichlet(alpha) allocation.

    alpha=0.5  → non-IID (heterogeneous clients, realistic FL deployment)
    alpha=100  → near-IID (each client sees roughly equal class proportions)

    Each class's samples are independently allocated across clients using
    Dirichlet proportions. The resulting per-client datasets are shuffled.

    Returns: list of (X_k, y_k) for k = 0 .. n_clients-1
    """
    clients_X = [[] for _ in range(n_clients)]
    clients_y = [[] for _ in range(n_clients)]

    for cls in range(N_CLASSES):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)

        props  = rng.dirichlet(alpha * np.ones(n_clients))
        counts = np.floor(props * len(idx)).astype(int)
        counts[-1] += len(idx) - counts.sum()    # assign remainder
        counts = np.maximum(counts, 0)

        start = 0
        for k in range(n_clients):
            end = start + counts[k]
            if counts[k] > 0:
                clients_X[k].append(X[idx[start:end]])
                clients_y[k].append(y[idx[start:end]])
            start = end

    result = []
    for k in range(n_clients):
        if clients_X[k]:
            Xk = np.concatenate(clients_X[k])
            yk = np.concatenate(clients_y[k])
            perm = rng.permutation(len(Xk))
            result.append((Xk[perm], yk[perm]))
        else:
            result.append((
                np.zeros((0, N_FEATURES), dtype=np.float32),
                np.zeros(0,              dtype=np.int64)
            ))
    return result


# ── Main dataset loader ───────────────────────────────────────────────────────

def load_ids_dataset(data_dir=DATA_DIR, n_clients=10, alpha=0.5,
                     max_per_class=5000, test_fraction=0.2, verbose=True):
    """
    Load CICIDS2017, build temporal concept stream, partition for FL.

    Steps:
      1. Load each day's CSV(s) with cleaning + subsampling
      2. Fit StandardScaler on ALL training data (before test split)
      3. Stratified 20% global test set (combined across all concepts)
      4. Dirichlet(alpha) partition training data per concept per client

    Returns:
        concepts       : list of dicts — {name, client_data, description}
        test_set       : (X_test, y_test) — global test, scaled
        concept_tests  : dict[name] → (X_test_c, y_test_c) — per-concept test
        scaler         : fitted StandardScaler
        feat_cols      : list of 78 feature column names
    """
    rng = np.random.default_rng(SEED)

    if verbose:
        print("Loading CICIDS2017 concepts...")

    raw_concepts = []

    if CONCEPT_MODE == '8session':
        concept_spec = [(n, f, d, h) for n, f, h, d in CONCEPT_FILES_8SESSION]
    else:
        base = (CONCEPT_FILES_IMPROVED if DATA_VERSION == 'improved'
                else CONCEPT_FILES)
        concept_spec = [(n, f, d, None) for n, f, d in base]
    if DATA_VERSION == 'improved' and data_dir == DATA_DIR:
        data_dir = DATA_DIR_IMPROVED
    for concept_name, files, desc, hours in concept_spec:
        X, y, feat_cols = load_concept_csv(files, data_dir, max_per_class, hours)
        if len(X) == 0:
            raise RuntimeError(f"concept '{concept_name}' loaded zero rows "
                               f"from {files} hours={hours}")
        unique, counts = np.unique(y, return_counts=True)
        dist = {CLASS_NAMES[u]: int(c) for u, c in zip(unique, counts)}
        if verbose:
            print(f"  {concept_name:12s}: {sum(counts):>7,} samples | {dist}")
        raw_concepts.append((concept_name, X, y, desc))

    # Split FIRST, then fit the scaler on training rows only.
    # This previously ran the other way round — the scaler was fit on each
    # concept's full array and the split was taken from those same arrays, so
    # test rows contributed to the mean/std. Small at this sample size, but it
    # is leakage and a reviewer would flag it.
    test_X_parts, test_y_parts = [], []
    train_parts = []

    for concept_name, X, y, desc in raw_concepts:
        if len(np.unique(y)) >= 2 and len(X) >= 20:
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_fraction,
                    stratify=y, random_state=SEED
                )
            except ValueError:   # some class has only 1 sample
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_fraction, random_state=SEED
                )
        else:
            n_te  = max(2, int(len(X) * test_fraction))
            idx_s = rng.choice(len(X), n_te, replace=False)
            idx_t = np.setdiff1d(np.arange(len(X)), idx_s)
            X_tr, X_te = X[idx_t], X[idx_s]
            y_tr, y_te = y[idx_t], y[idx_s]

        test_X_parts.append(X_te)
        test_y_parts.append(y_te)
        train_parts.append((concept_name, X_tr, y_tr, desc))

    if PREPROC == 'causal':
        # only what exists when the stream starts: the first concept's training rows
        fill = finite_max(train_parts[0][1])
        train_parts = [(n, fill_nonfinite(Xt, fill), yt, d) for n, Xt, yt, d in train_parts]
        test_X_parts = [fill_nonfinite(Xt, fill) for Xt in test_X_parts]
        fit_rows = train_parts[0][1]
    else:
        fit_rows = np.concatenate([X_tr for _, X_tr, _, _ in train_parts])
    scaler = StandardScaler()
    scaler.fit(fit_rows)
    if verbose:
        print(f"\nStandardScaler fitted on {len(fit_rows):,} training rows ({PREPROC})")

    # Scale test set
    X_test = scaler.transform(np.concatenate(test_X_parts))
    y_test = np.concatenate(test_y_parts)
    if verbose:
        print(f"Global test set: {len(X_test):,} samples")

    # Per-concept test sets (for tracking per-concept accuracy)
    concept_tests = {
        name: (scaler.transform(Xte), yte)
        for (name, *_), Xte, yte in zip(raw_concepts, test_X_parts, test_y_parts)
    }

    # Scale training data and Dirichlet-partition per concept
    concepts = []
    for concept_name, X_tr, y_tr, desc in train_parts:
        X_tr_sc = scaler.transform(X_tr)
        client_data = dirichlet_partition(X_tr_sc, y_tr, n_clients, alpha, rng)
        sizes = [len(c[1]) for c in client_data]
        if verbose:
            print(f"  {concept_name:12s}: clients have "
                  f"{min(sizes)}-{max(sizes)} samples (α={alpha})")
        concepts.append({
            'name':        concept_name,
            'description': desc,
            'client_data': client_data,   # list of (X_k, y_k)
            'test':        concept_tests[concept_name],   # this concept's held-out rows
        })

    if verbose:
        print(f"\nReady: {len(concepts)} concepts × {n_clients} clients "
              f"(Dirichlet α={alpha})\n")

    return concepts, (X_test, y_test), concept_tests, scaler, feat_cols


# ── Model: IDS DNN ────────────────────────────────────────────────────────────

class IDSDNN(nn.Module):
    """
    Feedforward DNN for tabular IDS classification.

    Input:  (B, N_FEATURES=78)  — normalized flow features
    Output: (B, N_CLASSES=7)    — class logits

    Architecture follows standard IDS literature:
      78 → 256 → 128 → 64 → 7, with BatchNorm + ReLU + Dropout per layer.
    BatchNorm is robust to growing memory sizes during continual training.
    """

    def __init__(self, n_features=N_FEATURES, n_classes=N_CLASSES, dropout=0.3,
                 norm=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            make_norm(256, norm),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            make_norm(128, norm),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            make_norm(64, norm),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)

    def get_confidence(self, x):
        """
        Max softmax probability = model's confidence in its prediction.
        Used by BetaDriftDetector: high conf → model is certain → no drift.
        Requires model to be in eval() mode before calling.
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
            return probs.max(dim=1).values


def count_parameters_ids(model=None):
    if model is None:
        model = IDSDNN()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] IDSDNN parameters: {n:,}")
    return n


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_ids(state_dict, X_test, y_test, device,
                 n_classes=N_CLASSES, verbose=False):
    """
    Evaluate a model state on (X_test, y_test).
    Returns (overall_accuracy, per_class_f1_dict, macro_f1, per_class_recall_dict).

    NOTE (fixed 2026-08-12): the third return value used to be per-class
    *recall* (accuracy over rows of that true class) while callers stored it
    under 'f1_<class>' keys. macro_f1 is the mean of per-class F1 by
    definition, so the old per-class table could never sum to it. Per-class F1
    is now computed properly; per-class recall is still returned separately
    because detection rate is a meaningful IDS metric in its own right, and it
    is what every previously published per-class table actually contains.
    """
    model = IDSDNN(n_classes=n_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    X_t = torch.tensor(X_test).float().to(device)
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).cpu().numpy()

    overall   = float(accuracy_score(y_test, preds))
    macro_f1  = float(f1_score(y_test, preds, average='macro',
                               zero_division=0))

    labels = list(range(n_classes))
    f1s    = f1_score(y_test, preds, average=None, labels=labels, zero_division=0)
    recs   = recall_score(y_test, preds, average=None, labels=labels, zero_division=0)

    present   = set(np.unique(y_test).tolist())
    per_class = {c: (float(f1s[i])  if i in present else float('nan'))
                 for i, c in enumerate(CLASS_NAMES)}
    per_recall = {c: (float(recs[i]) if i in present else float('nan'))
                  for i, c in enumerate(CLASS_NAMES)}

    if verbose:
        print(classification_report(y_test, preds,
              target_names=CLASS_NAMES, zero_division=0))

    return overall, per_class, macro_f1, per_recall
