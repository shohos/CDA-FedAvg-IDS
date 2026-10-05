"""
UNSW-NB15 Data Loader for CDA-FedAvg extension.

Inspection facts (from unsw_inspect.py output):
  - Pre-partitioned files: UNSW_NB15_training-set.csv (175,341 rows)
                            UNSW_NB15_testing-set.csv  (82,332 rows)
  - 45 columns total (with header)
  - id (col 0):            row ID — dropped
  - proto/service/state:   categorical → ordinal encoded → 3 integer features
  - 40 other numeric features
  - attack_cat (col 43):   10-class label (object)
  - label (col 44):        binary 0/1 — dropped (we use multiclass attack_cat)
  - No inf or NaN values in sample
  → N_FEATURES = 42  (39 numeric + 3 ordinal-encoded)
  → N_CLASSES  = 10  (Normal + 9 attack families)

Concept drift design — attack kill chain (5 stages, mirrors CICIDS2017 days):
  Concept 0: Normal         (baseline: no attacks)
  Concept 1: Reconnaissance + Analysis   (intelligence gathering)
  Concept 2: Exploits + Fuzzers          (active exploitation)
  Concept 3: DoS + Generic               (service disruption)
  Concept 4: Backdoors + Shellcode + Worms (persistence/malware)

Each concept includes Normal traffic + the new attack family,
identical to CICIDS2017 where BENIGN appeared in every concept.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
                             recall_score)

# ── Paths ─────────────────────────────────────────────────────────────────────

# ROOT = this file's directory, so the same code runs on Colab (Drive) and locally.
ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(ROOT, "CSV Files", "Training and Testing Sets")
TRAIN_CSV = "UNSW_NB15_training-set.csv"
TEST_CSV  = "UNSW_NB15_testing-set.csv"
CKPT_DIR  = os.path.join(ROOT, "checkpoints")
SEED      = 42

# ── Dataset constants ─────────────────────────────────────────────────────────

N_FEATURES_UNSW = 42     # 39 numeric + 3 ordinal-encoded (proto, service, state)
N_CLASSES_UNSW  = 10     # Normal + 9 attack families

# Categorical columns requiring ordinal encoding
CAT_COLS = ['proto', 'service', 'state']

# Column to drop (not features, not the multiclass label)
DROP_COLS = ['id', 'label']      # drop row-id and binary label; keep attack_cat

# ── Class definitions ─────────────────────────────────────────────────────────

# See ids_loader.NORM for the rationale. 'batch' reproduces published results.
NORM = 'layer'   # 'batch' | 'layer' — see ids_loader


def make_norm(dim, norm=None):
    norm = norm or NORM
    return nn.LayerNorm(dim) if norm == 'layer' else nn.BatchNorm1d(dim)


CLASS_NAMES_UNSW = [
    'Normal', 'Fuzzers', 'Analysis', 'Backdoors',
    'DoS', 'Exploits', 'Generic', 'Reconnaissance',
    'Shellcode', 'Worms'
]
CLASS_TO_IDX_UNSW = {c: i for i, c in enumerate(CLASS_NAMES_UNSW)}

# ── Concept groupings — attack kill chain ─────────────────────────────────────

UNSW_CONCEPTS = [
    (
        'Normal',
        ['Normal'],
        'Baseline normal traffic — no attacks visible yet'
    ),
    (
        'Reconnaissance',
        ['Reconnaissance', 'Analysis'],
        'Intelligence gathering phase'
    ),
    (
        'Exploitation',
        ['Exploits', 'Fuzzers'],
        'Active exploitation phase'
    ),
    (
        'Disruption',
        ['DoS', 'Generic'],
        'Service disruption phase'
    ),
    (
        'Persistence',
        ['Backdoors', 'Shellcode', 'Worms'],
        'Malware persistence/backdoor phase'
    ),
]

N_CONCEPTS_UNSW = len(UNSW_CONCEPTS)

# One concept per attack class, kill-chain ORDER preserved. Isolates two things
# the grouped spec confounds: whether a class is hurt by sharing a concept, and
# whether it is hurt by arriving early. Reconnaissance stays at index 1 here, so
# a recovery is attributable to un-pairing it from Analysis, not to position.
UNSW_CONCEPTS_PERCLASS = [
    ('Normal',         ['Normal'],         'Baseline normal traffic'),
    ('Reconnaissance', ['Reconnaissance'], 'Intelligence gathering'),
    ('Analysis',       ['Analysis'],       'Intelligence gathering'),
    ('Exploits',       ['Exploits'],       'Active exploitation'),
    ('Fuzzers',        ['Fuzzers'],        'Active exploitation'),
    ('DoS',            ['DoS'],            'Service disruption'),
    ('Generic',        ['Generic'],        'Service disruption'),
    ('Backdoors',      ['Backdoors'],      'Persistence'),
    ('Shellcode',      ['Shellcode'],      'Persistence'),
    ('Worms',          ['Worms'],          'Persistence'),
]

# 'killchain' (5 grouped stages, default) | 'perclass' (10, one class each)
CONCEPT_MODE = 'killchain'

# Preprocessing policy (audit F1): see ids_loader.PREPROC. 'causal' fits the
# ordinal encoder and the scaler on the first concept's training rows only;
# categories first seen later become -1. 'all_train' reproduces the original runs.
# The concepts are groups of attack families in kill-chain order, built for this
# study; UNSW-NB15's own capture chronology is not used.
PREPROC = 'causal'   # 'causal' | 'all_train'


def concept_spec():
    return (UNSW_CONCEPTS_PERCLASS if CONCEPT_MODE == 'perclass'
            else UNSW_CONCEPTS)



# ── Label normalisation ───────────────────────────────────────────────────────

def normalize_unsw_label(raw: str):
    """
    Map raw attack_cat string to canonical CLASS_NAMES_UNSW entry.
    Handles:
      - Empty string / whitespace-only / NaN → 'Normal' (label=0 records)
      - Trailing/leading spaces              → stripped
      - 'Backdoor' (singular) as well as 'Backdoors' (plural) — both present in dataset
      - All 9 attack families
    Returns None for any unrecognised value (will be dropped).
    """
    l = str(raw).strip()
    if l == '' or l == 'nan' or l.lower() == 'normal':
        return 'Normal'
    # Exact match (most common path)
    if l in CLASS_NAMES_UNSW:
        return l
    # 'Backdoor' singular → 'Backdoors' (training file uses singular form)
    if l == 'Backdoor':
        return 'Backdoors'
    # Case-insensitive fallback
    l_lower = l.lower()
    for cls in CLASS_NAMES_UNSW:
        if l_lower == cls.lower():
            return cls
    # Singular/plural fallback (handles any other s-discrepancy)
    for cls in CLASS_NAMES_UNSW:
        if l_lower.rstrip('s') == cls.lower().rstrip('s'):
            return cls
    return None    # unknown — drop


# ── Load and preprocess pre-partitioned files ─────────────────────────────────

def load_unsw_raw(csv_path: str) -> pd.DataFrame:
    """
    Load one UNSW-NB15 CSV file, normalise attack_cat, drop unwanted columns.
    Returns DataFrame with all feature columns + 'attack_cat' (canonical).
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalise attack_cat
    df['attack_cat'] = df['attack_cat'].apply(normalize_unsw_label)
    df = df[df['attack_cat'].notna()].reset_index(drop=True)

    # Drop id and binary label
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    return df


def build_feature_matrix(df: pd.DataFrame,
                          ordinal_encoder=None,
                          fit_encoder: bool = False):
    """
    Convert DataFrame to (X, y, feat_cols).
    Ordinal-encodes proto/service/state; StandardScaler is applied separately.

    Args:
        df:              input DataFrame with 'attack_cat' column
        ordinal_encoder: fitted OrdinalEncoder (None → create new one)
        fit_encoder:     if True, fit the encoder on df's categorical columns

    Returns:
        X:               float32 (N, N_FEATURES_UNSW)
        y:               int64   (N,) class indices
        feat_cols:       list of 43 column names in order
        ordinal_encoder: fitted or passed-through encoder
    """
    # Separate label
    y_str = df['attack_cat'].values
    feat_df = df.drop(columns=['attack_cat'])

    # Ordinal-encode categorical columns
    cat_present = [c for c in CAT_COLS if c in feat_df.columns]
    if cat_present:
        if ordinal_encoder is None:
            ordinal_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
        if fit_encoder:
            feat_df[cat_present] = ordinal_encoder.fit_transform(
                feat_df[cat_present]
            )
        else:
            feat_df[cat_present] = ordinal_encoder.transform(
                feat_df[cat_present]
            )

    feat_cols = list(feat_df.columns)
    assert len(feat_cols) == N_FEATURES_UNSW, \
        f"Expected {N_FEATURES_UNSW} features, got {len(feat_cols)}. " \
        f"Columns: {feat_cols}"

    X = feat_df.values.astype(np.float32)
    y = np.array([CLASS_TO_IDX_UNSW[c] for c in y_str], dtype=np.int64)

    return X, y, feat_cols, ordinal_encoder


# ── Dirichlet partitioning (identical to CICIDS2017 loader) ──────────────────

def dirichlet_partition_unsw(X, y, n_clients, alpha, rng):
    """Partition (X, y) across n_clients via Dirichlet(alpha) allocation."""
    clients_X = [[] for _ in range(n_clients)]
    clients_y = [[] for _ in range(n_clients)]

    for cls in range(N_CLASSES_UNSW):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        props  = rng.dirichlet(alpha * np.ones(n_clients))
        counts = np.floor(props * len(idx)).astype(int)
        counts[-1] += len(idx) - counts.sum()
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
                np.zeros((0, N_FEATURES_UNSW), dtype=np.float32),
                np.zeros(0, dtype=np.int64)
            ))
    return result


# ── Main loader ───────────────────────────────────────────────────────────────

def load_unsw_dataset(data_dir=DATA_DIR, n_clients=10, alpha=0.5,
                      max_per_class=5000, verbose=True):
    """
    Load UNSW-NB15 pre-partitioned files, build temporal concept stream.

    Uses the existing train/test split (no manual 80/20 split needed):
      Training: UNSW_NB15_training-set.csv  → concept stream + Dirichlet partition
      Testing:  UNSW_NB15_testing-set.csv   → fixed global test set

    Returns:
        concepts       : list of dicts {name, client_data, description}
        test_set       : (X_test, y_test) — scaled
        scaler         : fitted StandardScaler (on training data only)
        feat_cols      : list of 43 feature names
    """
    rng = np.random.default_rng(SEED)

    # ── Load raw files ────────────────────────────────────────────────────────
    if verbose:
        print("Loading UNSW-NB15 pre-partitioned files...")

    train_path = os.path.join(data_dir, TRAIN_CSV)
    test_path  = os.path.join(data_dir, TEST_CSV)

    df_train = load_unsw_raw(train_path)
    df_test  = load_unsw_raw(test_path)

    # ── Sanity check: training should be larger than testing ─────────────────
    # If they are identical in size / distribution, the train file likely
    # resolved to the test file (Colab import cache / path issue).
    # Fallback: combine both, then stratified 80/20 split.
    if len(df_train) == len(df_test) and \
            set(df_train['attack_cat'].unique()) == set(df_test['attack_cat'].unique()) and \
            abs(len(df_train) - len(df_test)) < 100:
        if verbose:
            print(f"  [WARN] Training and testing files appear identical "
                  f"({len(df_train):,} rows each). "
                  f"Falling back to 80/20 stratified split.")
        from sklearn.model_selection import train_test_split as tts
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        # Stratified split on attack_cat
        train_idx, test_idx = tts(
            np.arange(len(df_combined)),
            test_size=0.2,
            stratify=df_combined['attack_cat'].values,
            random_state=SEED
        )
        df_train = df_combined.iloc[train_idx].reset_index(drop=True)
        df_test  = df_combined.iloc[test_idx].reset_index(drop=True)
        if verbose:
            print(f"  [INFO] Fallback split: train={len(df_train):,}, "
                  f"test={len(df_test):,}")

    if verbose:
        dist_tr = df_train['attack_cat'].value_counts().to_dict()
        dist_te = df_test['attack_cat'].value_counts().to_dict()
        print(f"  Training:  {len(df_train):,} rows | {dist_tr}")
        print(f"  Testing:   {len(df_test):,} rows  | {dist_te}")

    # ── Encode and build feature matrices ────────────────────────────────────
    if PREPROC == 'causal':
        # encoder and scaler are fitted below, on the first concept's rows only
        y_train_all = np.array([CLASS_TO_IDX_UNSW[c] for c in df_train['attack_cat']],
                               dtype=np.int64)
        X_train_scaled = None
    else:
        X_train_all, y_train_all, feat_cols, ordinal_enc = build_feature_matrix(
            df_train, fit_encoder=True
        )
        X_test_all, y_test_all, _, _ = build_feature_matrix(
            df_test, ordinal_encoder=ordinal_enc, fit_encoder=False
        )

        # ── Fit StandardScaler on training data only ──────────────────────────
        scaler = StandardScaler()
        scaler.fit(X_train_all)
        X_train_scaled = scaler.transform(X_train_all)
        X_test_scaled  = scaler.transform(X_test_all)

        if verbose:
            print(f"  Scaler fitted on {len(X_train_all):,} training samples")
            print(f"  Test set: {len(X_test_scaled):,} samples (pre-partitioned, no split needed)")

    # ── Build concept groups from training data ───────────────────────────────
    # Assign concept index to each training sample based on attack_cat
    spec = concept_spec()
    # Normal recurs in EVERY concept, so more concepts would silently inflate the
    # pooled Normal count and move the IID ceiling. Scale its per-concept cap so
    # the pooled dataset is identical across modes; attack classes appear in
    # exactly one concept either way and keep the full cap.
    normal_cap = max(1, max_per_class * len(UNSW_CONCEPTS) // len(spec))         if max_per_class is not None else None
    concept_map = {}   # class_name → concept_index
    for c_idx, (c_name, attack_cats, _) in enumerate(spec):
        for cat in attack_cats:
            concept_map[cat] = c_idx
        concept_map['Normal'] = 0   # Normal always goes to concept 0

    if verbose:
        print(f"\nBuilding {N_CONCEPTS_UNSW} temporal concepts (attack kill chain)...")

    concepts = []
    for c_idx, (c_name, attack_cats, desc) in enumerate(spec):

        # Select training rows for this concept:
        # Concept 0 = Normal only
        # Concepts 1-4 = Normal + their specific attack families
        if c_idx == 0:
            include_cats = {'Normal'}
        else:
            include_cats = set(attack_cats) | {'Normal'}

        # Boolean mask on original (unscaled) label array
        mask = np.isin(
            [CLASS_NAMES_UNSW[yi] for yi in y_train_all],
            list(include_cats)
        )
        rows = np.where(mask)[0]          # row numbers in the training file
        y_c = y_train_all[mask]

        # Per-class subsampling to keep dataset manageable
        if max_per_class is not None:
            keep = []
            for cls_idx in range(N_CLASSES_UNSW):
                idx = np.where(y_c == cls_idx)[0]
                if len(idx) == 0:
                    continue
                cap = (normal_cap if CLASS_NAMES_UNSW[cls_idx] == 'Normal'
                       else max_per_class)
                if len(idx) > cap:
                    idx = rng.choice(idx, cap, replace=False)
                keep.append(idx)
            if keep:
                keep = np.sort(np.concatenate(keep))
                rows, y_c = rows[keep], y_c[keep]

        if X_train_scaled is None:        # 'causal', first concept: fit on it alone
            ordinal_enc = OrdinalEncoder(handle_unknown='use_encoded_value',
                                         unknown_value=-1)
            ordinal_enc.fit(df_train.iloc[rows][CAT_COLS])
            X_train_all, _, feat_cols, _ = build_feature_matrix(df_train, ordinal_enc)
            X_test_all, y_test_all, _, _ = build_feature_matrix(df_test, ordinal_enc)
            scaler = StandardScaler().fit(X_train_all[rows])
            X_train_scaled = scaler.transform(X_train_all)
            X_test_scaled = scaler.transform(X_test_all)
            if verbose:
                print(f"  Encoder and scaler fitted on {len(rows):,} rows of "
                      f"'{c_name}' (causal)")
        X_c = X_train_scaled[rows]

        unique, counts = np.unique(y_c, return_counts=True)
        dist = {CLASS_NAMES_UNSW[u]: int(c) for u, c in zip(unique, counts)}
        if verbose:
            print(f"  {c_name:15s}: {len(X_c):>7,} samples | {dist}")

        # Dirichlet partition across clients
        client_data = dirichlet_partition_unsw(X_c, y_c, n_clients, alpha, rng)
        sizes = [len(ck[1]) for ck in client_data]
        if verbose:
            print(f"               clients: {min(sizes)}-{max(sizes)} samples (α={alpha})")

        te = np.isin(y_test_all, [CLASS_TO_IDX_UNSW[c] for c in include_cats])
        concepts.append({
            'name':        c_name,
            'description': desc,
            'client_data': client_data,
            'test':        (X_test_scaled[te], y_test_all[te]),   # official test rows
        })                                                      # of this concept's classes

    test_set = (X_test_scaled, y_test_all)
    if verbose:
        print(f"\nReady: {N_CONCEPTS_UNSW} concepts × {n_clients} clients (Dirichlet α={alpha})")

    return concepts, test_set, scaler, feat_cols


# ── Model ─────────────────────────────────────────────────────────────────────

class UNSW_DNN(nn.Module):
    """
    Feedforward DNN for UNSW-NB15 tabular classification.
    Input:  (B, N_FEATURES_UNSW=42)  — normalized flow features
    Output: (B, N_CLASSES_UNSW=10)   — class logits
    Same architecture as IDSDNN but adapted for 43 inputs / 10 outputs.
    """

    def __init__(self, n_features=N_FEATURES_UNSW,
                 n_classes=N_CLASSES_UNSW, dropout=0.3, norm=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            make_norm(256, norm), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            make_norm(128, norm), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            make_norm(64, norm),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)

    def get_confidence(self, x):
        """Max softmax probability — used by BetaDriftDetector."""
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1).max(dim=1).values


def count_parameters_unsw(model=None):
    if model is None:
        model = UNSW_DNN()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] UNSW_DNN parameters: {n:,}")
    return n


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_unsw(state_dict, X_test, y_test, device,
                  n_classes=N_CLASSES_UNSW, verbose=False):
    """
    Accuracy + per-class F1 + macro F1 + per-class recall on UNSW-NB15 test set.
    See evaluate_ids for the note on why per-class F1 replaced per-class recall.
    """
    model = UNSW_DNN(n_classes=n_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    X_t   = torch.tensor(X_test).float().to(device)
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).cpu().numpy()

    overall  = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average='macro', zero_division=0))

    labels = list(range(n_classes))
    f1s    = f1_score(y_test, preds, average=None, labels=labels, zero_division=0)
    recs   = recall_score(y_test, preds, average=None, labels=labels, zero_division=0)

    present    = set(np.unique(y_test).tolist())
    per_class  = {c: (float(f1s[i])  if i in present else float('nan'))
                  for i, c in enumerate(CLASS_NAMES_UNSW)}
    per_recall = {c: (float(recs[i]) if i in present else float('nan'))
                  for i, c in enumerate(CLASS_NAMES_UNSW)}

    if verbose:
        print(classification_report(y_test, preds,
              target_names=CLASS_NAMES_UNSW, zero_division=0))

    return overall, per_class, macro_f1, per_recall
