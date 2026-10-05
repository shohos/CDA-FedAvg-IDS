"""
IDS CDA-FedAvg Pipeline — Novel Thesis Contribution.

Runs three experiments on CICIDS2017 using Dirichlet non-IID partitioning:
  Experiment A: FedAvg IID           (upper bound — shuffled data)
  Experiment B: FedAvg Non-stationary (baseline — temporal order, no memory)
  Experiment C: CDA-FedAvg IDS       (main contribution — drift-aware rehearsal)

Concept drift design:
  Each CICIDS2017 day = one concept (mirrors HAR phone positions):
    Monday    → BENIGN baseline
    Tuesday   → BruteForce attacks emerge
    Wednesday → DoS attacks emerge
    Thursday  → WebAttack attacks emerge
    Friday    → DDoS + PortScan + Bot attacks emerge

Federated setup:
  10 clients, Dirichlet(α=0.5) non-IID partitioning per concept
  Global 20% stratified test set for evaluation

All experiments use identical SGD (lr=0.01, no momentum, clip_norm=10.0)
so the comparison is attributable solely to the CDA memory mechanism.
"""

import numpy as np
import torch
import torch.nn as nn
import os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ids_loader          # imported as a module so SEED is read at call time,
                           # not frozen at import (baseline_v2 rebinds it per seed)
from ids_loader import (
    IDSDNN, load_ids_dataset, evaluate_ids, count_parameters_ids,
    CLASS_NAMES, N_CLASSES, N_FEATURES, CKPT_DIR, SEED,
    CONCEPT_FILES, N_CONCEPTS
)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Experiment settings ───────────────────────────────────────────────────────

N_CLIENTS        = 10
ALPHA            = 0.5      # Dirichlet concentration (non-IID)
MAX_PER_CLASS    = 5000     # subsample cap per class per concept (memory)
TEST_FRACTION    = 0.2
N_ROUNDS         = 25       # total FL rounds (5 concepts × 5 rounds each)
R_PER_CONCEPT    = N_ROUNDS // N_CONCEPTS   # = 5 rounds per concept
LOCAL_EPOCHS     = 10
BATCH_SIZE       = 100
SGD_LR           = 0.01
CLIP_NORM        = 10.0     # gradient clipping (same as HAR Step 5)

QUICK_TEST       = True     # True → small data, fast verify run (~30s)

# ── CDA-FedAvg paper constants ────────────────────────────────────────────────

LAM         = 0.05    # λ: drift sensitivity
DELTA       = 100     # Δ: minimum detector sub-window
N_MAX       = 1000    # Nmax: sliding window max size
L_CAPACITY  = 1400    # L: memory budget
PER_CLASS   = L_CAPACITY // (2 * N_CLASSES)   # = 100 per class per concept


# ── Algorithm 5: BetaDriftDetector ───────────────────────────────────────────

from scipy.stats import beta as beta_dist
from drift_channels import HybridDriftDetector, train_ae, recon_scores

class BetaDriftDetector:
    """
    Casado et al. Algorithm 5 (driftDetection).

    Fixed 2026-08-13. The previous implementation computed a pooled-vs-split
    generalised likelihood ratio over the whole window,
        ll(Q_a|θ_a) + ll(Q_b|θ_b) − ll(Q|θ_pooled),
    which is a nested-model comparison and non-negative by construction. The
    paper (Alg. 5 line 12) accumulates over the RECENT sub-window only, as a
    ratio of the two fitted alternatives:
        s_k = Σ_{i=k+1..N} log[ f(q_i | α_a, β_a) / f(q_i | α_b, β_b) ].
    See FAITHFULNESS_AUDIT.md; detector_faithfulness.py measures the difference
    (small in practice — the gate decides, not the statistic).

    Each instance owns its RNG so the stochastic check cannot couple clients or
    leak state between experiments.
    """

    def __init__(self, window_size=N_MAX, delta=DELTA, lam=LAM, rng=None):
        self.window_size = window_size
        self.delta       = delta
        self.lam         = lam
        self.threshold   = -np.log(lam)
        self.window      = []
        self.rng         = rng if rng is not None else np.random.default_rng()

    def should_check(self, conf: float) -> bool:
        """Algorithm 4 line 12: run detection with probability e^{-2q}."""
        return float(np.exp(-2.0 * conf)) >= self.rng.random()

    def update(self, conf: float):
        self.window.append(float(np.clip(conf, 1e-6, 1 - 1e-6)))
        if len(self.window) > self.window_size:
            self.window.pop(0)

    @staticmethod
    def _fit_beta(x):
        x      = np.clip(x, 1e-6, 1 - 1e-6)
        mu, var = np.mean(x), np.var(x)
        max_v   = mu * (1 - mu) - 1e-6
        var     = np.clip(var, 1e-8, max_v if max_v > 1e-8 else 1e-6)
        f       = mu * (1 - mu) / var - 1
        return max(mu * f, 1e-6), max((1 - mu) * f, 1e-6)

    def _ll(self, x, a, b):
        return float(np.sum(beta_dist.logpdf(np.clip(x, 1e-6, 1-1e-6), a, b)))

    def detect(self, return_score=False):
        """Algorithm 5. Returns True if a drift is detected."""
        w = np.array(self.window)
        N = len(w)
        if N < 2 * self.delta:
            return (False, 0.0) if return_score else False
        sf = 0.0                                        # Alg. 5 line 1
        for k in range(self.delta, N - self.delta + 1):  # line 4
            qb, qa = w[:k], w[k:]                        # lines 5-6
            if np.mean(qa) > (1 - self.lam) * np.mean(qb):   # line 7
                continue
            ab, bb = self._fit_beta(qb)                  # line 9
            aa, ba = self._fit_beta(qa)                  # line 10
            qa_c = np.clip(qa, 1e-6, 1 - 1e-6)
            sk = float(np.sum(beta_dist.logpdf(qa_c, aa, ba)
                              - beta_dist.logpdf(qa_c, ab, bb)))  # lines 11-13
            sf = max(sf, sk)                             # line 14
        fired = bool(sf > self.threshold)                      # line 17
        return (fired, sf) if return_score else fired

    def reset(self):
        self.window.clear()

    def __len__(self):
        return len(self.window)


# ── Long-Term Memory (Algorithm 6) ───────────────────────────────────────────

QUORUM_MODE = 'count'   # 'count' = QUORUM clients must fire (Casado's rule is
                        # QUORUM=1). 'weight' = the clients that fired must
                        # collectively hold >= QUORUM_WEIGHT of the concept's
                        # data. Measured motivation: at alpha=0.1 the top 3
                        # clients hold 79-99% of a new concept's samples, so
                        # requiring 5 of 10 is impossible; at alpha=100 they
                        # hold 32-35%, so 3 firing is weak evidence. Weighting
                        # by n_k — which FedAvg already communicates — adapts to
                        # both without a per-alpha constant.
                        #
                        # TESTED AND REJECTED (null_control.py, alpha=0.1):
                        # weighting is actively harmful under extreme
                        # heterogeneity — false alarms 83% (CICIDS) / 100%
                        # (UNSW) against detection of 33% / 25%, i.e. it fires
                        # MORE on a no-drift stream than on real drift. The same
                        # concentration that lets 3 clients clear 50% on genuine
                        # drift lets 1 client clear it on a false alarm. Keep
                        # 'count'. Retained only so the result is reproducible.
QUORUM_WEIGHT = 0.5
QUORUM = 1   # clients that must fire before adaptation is triggered.
             # 1 = Casado's any-one rule. null_control.py shows the any-one
             # rule destroys the channel's client-level signal: 2.2 clients per
             # concept fire on a no-drift stream vs 7.25 on real drift, so a
             # threshold in between separates them.

OPTIMIZER = 'momentum'  # best measured (opt_sweep.py): momentum@0.01 wins on
                        # both datasets and both norms. 'sgd' = published runs.
                        # was 'sgd'  # 'sgd' | 'momentum' | 'adam' | 'adamw'
MOMENTUM  = 0.9      # used by 'momentum'
                     # SGD(lr=0.01, no momentum) was inherited from Casado's HAR
                     # CNN and carried unexamined into a different architecture,
                     # data regime and normalisation. Note for FL: client
                     # optimiser state is rebuilt every round (a fresh model is
                     # constructed per local_train call), so Adam's moments reset
                     # each round — the known limitation that motivates
                     # server-side adaptivity (Reddi et al., ICLR 2021).


def _make_opt(params, lr):
    if OPTIMIZER == 'momentum':
        return torch.optim.SGD(params, lr=lr, momentum=MOMENTUM)
    if OPTIMIZER == 'adam':
        return torch.optim.Adam(params, lr=lr)
    if OPTIMIZER == 'adamw':
        return torch.optim.AdamW(params, lr=lr)
    return torch.optim.SGD(params, lr=lr)


LOCAL_NORM = False   # FedBN (ICLR 2021): keep normalisation layers local instead
                     # of averaging them. Isolates WHY LayerNorm helped — we
                     # claim running-statistic averaging is the culprit, but
                     # LayerNorm changes two things at once (no running stats
                     # AND per-sample normalisation). BatchNorm+LOCAL_NORM has
                     # running stats but never averages them, so it separates
                     # the two explanations.
_LOCAL_NORM_STATE = {}


def _norm_keys(model):
    ks = set()
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.LayerNorm)):
            ks.update(f"{name}.{p}" for p, _ in mod.named_parameters())
            ks.update(f"{name}.{b}" for b, _ in mod.named_buffers())
    return ks


MEMORY_RULE = 'cap'   # 'cap' (ours, default) | 'floor' (Casado Alg. 6, faithful)
MEMORY_BOUND = True   # False reproduces the original runs, where the per-class cap
                      # applied per concept and nothing was evicted, so memory grew
                      # with the stream (audit F5). True: at most L_CAPACITY rows.

from memory_bound import evict_to_cap
import aggregators           # server rule for the aggregator study (FedAvg by default)


def _floor_prefix(y_c, n_classes, per_class):
    """
    Casado Alg. 6 line 1: keep collecting from the stream until every class
    present has at least PER_CLASS examples (or its supply is exhausted).
    Returns the length of the shortest prefix satisfying that condition.
    """
    present = [c for c in range(n_classes) if np.any(y_c == c)]
    need = {c: min(per_class, int(np.sum(y_c == c))) for c in present}
    have = {c: 0 for c in present}
    for i, yi in enumerate(y_c):
        have[int(yi)] = have.get(int(yi), 0) + 1
        if all(have[c] >= need[c] for c in present):
            return i + 1
    return len(y_c)


class LongTermMemory:
    """
    Algorithm 6 long-term memory.

    Two collection rules, selected by MEMORY_RULE:

      'cap'   — take AT MOST PER_CLASS per class per concept. What this project
                has always done, and what every published result used.
      'floor' — Casado's actual rule: L/(2M) is a MINIMUM, so keep collecting
                from the stream until every present class reaches it. Faithful
                to the paper, but built for HAR where each concept holds all
                classes in roughly equal proportion. On severely imbalanced IDS
                concepts it drags in a large majority-class tail.

    Default is 'cap' so existing results reproduce exactly. See
    FAITHFULNESS_AUDIT.md.
    """

    def __init__(self, rng=None):
        self.concepts = []   # list of (X, y, concept_name)
        self.rng = rng       # eviction draws (MEMORY_BOUND); per client, never global

    def collect_and_expand(self, X_c, y_c, name):
        """Collect this concept's contribution to L and append it."""
        if MEMORY_RULE == 'floor':
            n = _floor_prefix(y_c, N_CLASSES, PER_CLASS)
            if n > 0:
                self.concepts.append((X_c[:n], y_c[:n], name))
        else:
            Xs, ys = [], []
            for cls in range(N_CLASSES):
                idx = np.where(y_c == cls)[0]
                if len(idx) == 0:
                    continue
                n   = min(len(idx), PER_CLASS)
                Xs.append(X_c[idx[:n]])
                ys.append(y_c[idx[:n]])
            if Xs:
                self.concepts.append((
                    np.concatenate(Xs), np.concatenate(ys), name
                ))
        if MEMORY_BOUND:
            self.concepts = evict_to_cap(self.concepts, L_CAPACITY,
                                         self.rng or np.random.default_rng(0))

    def get_all(self):
        """Return full memory (X, y) for training."""
        if not self.concepts:
            return None, None
        return (np.concatenate([c[0] for c in self.concepts]),
                np.concatenate([c[1] for c in self.concepts]))

    def total_samples(self):
        return sum(len(c[1]) for c in self.concepts)

    def n_concepts(self):
        return len(self.concepts)


# ── Shared FL primitives ──────────────────────────────────────────────────────

def _metrics(per_class, per_recall):
    """Flatten per-class F1 and per-class recall into result-row keys."""
    return {**{f'f1_{c}':  per_class.get(c,  float('nan')) for c in CLASS_NAMES},
            **{f'rec_{c}': per_recall.get(c, float('nan')) for c in CLASS_NAMES}}


def fedavg_aggregate(updates):
    """Weighted FedAvg: w_global = Σ (n_k / N) * w_k. Other rules: aggregators.AGG."""
    if aggregators.AGG != 'fedavg':
        return aggregators.aggregate(updates, _weighted_mean)
    return _weighted_mean(updates)


def _weighted_mean(updates):
    total = sum(n for _, n in updates)
    agg   = {}
    for state, n in updates:
        w = n / total
        for k, v in state.items():
            agg[k] = agg.get(k, 0) + w * v.float()
    return {k: v for k, v in agg.items()}


def local_train_ids(global_state_gpu, X_gpu, y_gpu, device, client_id=None,
                    E=None, B=None,
                    lr=None, n_classes=N_CLASSES):
    """
    One local training round — SGD + gradient clipping.
    X_gpu: (N, N_FEATURES) float tensor already on device.
    """
    lr = SGD_LR if lr is None else lr
    E  = LOCAL_EPOCHS if E is None else E
    B  = BATCH_SIZE if B is None else B
    model = IDSDNN(n_classes=n_classes).to(device)
    if LOCAL_NORM and client_id is not None:
        st = dict(global_state_gpu)
        prev = _LOCAL_NORM_STATE.get(client_id)
        if prev:
            st.update({k: v.to(device) for k, v in prev.items()})
        model.load_state_dict(st)
    else:
        model.load_state_dict(global_state_gpu)
    model.train()

    n = len(X_gpu)
    # BatchNorm1d requires > 1 sample per batch.
    # Skip entirely if too few samples to form even one valid batch.
    if n < 2:
        return {k: v.cpu() for k, v in model.state_dict().items()}, n

    opt  = _make_opt(model.parameters(), lr)
    crit = nn.CrossEntropyLoss()
    hook = aggregators.local_hook(model, client_id, X_gpu, y_gpu, crit, B)   # FedProx/SCAFFOLD

    for _ in range(E):
        for start in range(0, n, B):
            Xb = X_gpu[start:start + B]
            yb = y_gpu[start:start + B]
            if len(Xb) < 2:   # last mini-batch might be exactly 1 sample
                continue
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            if hook:
                loss = loss + hook.penalty()
            loss.backward()
            if hook:
                hook.correct()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
    if hook:
        hook.finish()

    out = {k: v.cpu() for k, v in model.state_dict().items()}
    if LOCAL_NORM and client_id is not None:
        _LOCAL_NORM_STATE[client_id] = {k: out[k].clone()
                                        for k in _norm_keys(model) if k in out}
    return out, n


# ── Experiment A: FedAvg IID ──────────────────────────────────────────────────

def run_iid_fedavg(concepts, test_set, device, n_classes=N_CLASSES, verbose=True):
    """
    FedAvg baseline with IID data: all concept data pooled and shuffled.
    Each round: each client trains on their random share of the full dataset.
    Shows what FL can achieve without concept drift.
    """
    print("\n── Experiment A: FedAvg IID ─────────────────────────────")

    # Pool all concept data per client and shuffle (IID)
    rng = np.random.default_rng(ids_loader.SEED)
    all_X = [[] for _ in range(N_CLIENTS)]
    all_y = [[] for _ in range(N_CLIENTS)]
    for concept in concepts:
        for k, (Xk, yk) in enumerate(concept['client_data']):
            if len(Xk) > 0:
                all_X[k].append(Xk)
                all_y[k].append(yk)

    client_gpu = []
    for k in range(N_CLIENTS):
        if all_X[k]:
            Xk = np.concatenate(all_X[k])
            yk = np.concatenate(all_y[k])
            if len(Xk) < 2:          # skip clients too small for BatchNorm
                client_gpu.append(None)
                continue
            perm = rng.permutation(len(Xk))
            Xk, yk = Xk[perm], yk[perm]
            client_gpu.append((
                torch.tensor(Xk).float().to(device),
                torch.tensor(yk).long().to(device)
            ))
        else:
            client_gpu.append(None)

    global_state_gpu = IDSDNN(n_classes=n_classes).to(device).state_dict()
    global_state_cpu = {k: v.cpu() for k, v in global_state_gpu.items()}
    results = []

    for r in range(1, N_ROUNDS + 1):
        updates = []
        for ci, (Xg, yg) in [(i, c) for i, c in enumerate(client_gpu) if c is not None]:
            state_cpu, n = local_train_ids(global_state_gpu, Xg, yg, device, client_id=ci)
            updates.append((state_cpu, n))
        if updates:
            global_state_cpu = fedavg_aggregate(updates)
            global_state_gpu = {k: v.to(device) for k, v in global_state_cpu.items()}

        overall, per_class, macro_f1, per_recall = evaluate_ids(
            global_state_cpu, test_set[0], test_set[1], device
        )
        results.append({'round': r, 'overall': overall, 'macro_f1': macro_f1,
                        **_metrics(per_class, per_recall)})
        if r % 5 == 0 and verbose:
            print(f"  Round {r:2d}/25 | Acc={overall:.4f} | MacroF1={macro_f1:.4f}")

    if verbose:
        print(f"  [IID FedAvg] Final: Acc={results[-1]['overall']:.4f} "
              f"MacroF1={results[-1]['macro_f1']:.4f}")
    return global_state_cpu, results


# ── Experiment A2: FedAvg IID-stream (compute-matched ceiling) ───────────────

def run_iid_stream_fedavg(concepts, test_set, device, n_classes=N_CLASSES,
                          verbose=True):
    """
    IID upper bound with the SAME training budget as non-stationary FedAvg.

    run_iid_fedavg retrains on each client's whole pooled shard every round, so
    it performs ~19x the gradient steps of the non-stationary baseline — the
    IID-vs-non-stationary gap therefore mixes catastrophic forgetting with a
    large optimisation advantage.

    This variant pools each client's concepts, shuffles (destroying the temporal
    ordering — the variable under study), and streams the result in N_ROUNDS
    equal slices, one per round. Non-stationary streams 5 concepts x 5 slices =
    25 slices of the same total data, so the two are step-for-step matched and
    the only remaining difference is concept ordering.
    """
    if verbose:
        print("\n── Experiment A2: FedAvg IID-stream (compute-matched) ───")

    rng = np.random.default_rng(ids_loader.SEED)
    client_slices = []
    for k in range(N_CLIENTS):
        Xs = [Xk for c in concepts for Xk, _ in [c['client_data'][k]] if len(Xk)]
        ys = [yk for c in concepts for _, yk in [c['client_data'][k]] if len(yk)]
        if not Xs:
            client_slices.append([(None, None)] * N_ROUNDS)
            continue
        Xk, yk = np.concatenate(Xs), np.concatenate(ys)
        perm = rng.permutation(len(Xk))          # <- kills concept ordering
        Xk, yk = Xk[perm], yk[perm]
        client_slices.append([
            (Xk[sp], yk[sp]) if len(sp) > 0 else (None, None)
            for sp in np.array_split(np.arange(len(Xk)), N_ROUNDS)
        ])

    global_state_gpu = IDSDNN(n_classes=n_classes).to(device).state_dict()
    global_state_cpu = {k: v.cpu() for k, v in global_state_gpu.items()}
    results = []

    for r in range(1, N_ROUNDS + 1):
        updates = []
        for ci, k_slices in enumerate(client_slices):
            Xsl, ysl = k_slices[r - 1]
            if Xsl is None or len(Xsl) == 0:
                continue
            state_cpu, n = local_train_ids(
                global_state_gpu,
                torch.tensor(Xsl).float().to(device),
                torch.tensor(ysl).long().to(device),
                device, client_id=ci
            )
            updates.append((state_cpu, n))
        if updates:
            global_state_cpu = fedavg_aggregate(updates)
            global_state_gpu = {k: v.to(device) for k, v in global_state_cpu.items()}

        overall, per_class, macro_f1, per_recall = evaluate_ids(
            global_state_cpu, test_set[0], test_set[1], device
        )
        results.append({'round': r, 'overall': overall, 'macro_f1': macro_f1,
                        **_metrics(per_class, per_recall)})
        if r % 5 == 0 and verbose:
            print(f"  Round {r:2d}/25 | Acc={overall:.4f} | MacroF1={macro_f1:.4f}")

    if verbose:
        print(f"  [IID-stream] Final: Acc={results[-1]['overall']:.4f} "
              f"MacroF1={results[-1]['macro_f1']:.4f}")
    return global_state_cpu, results


# ── Experiment B: FedAvg Non-Stationary ──────────────────────────────────────

def run_nonstationary_fedavg(concepts, test_set, device, n_classes=N_CLASSES, verbose=True,
                             on_concept_start=None):
    """
    FedAvg with temporally-ordered data and NO memory.
    5 rounds per concept, each round trains on ~1/5 of that concept's data.
    Catastrophic forgetting expected as new attack types overwrite old knowledge.
    on_concept_start(i, state): called before the model sees concept i >= 1
    (forward-stream evaluation).
    """
    print("\n── Experiment B: FedAvg Non-Stationary ──────────────────")

    global_state_gpu = IDSDNN(n_classes=n_classes).to(device).state_dict()
    global_state_cpu = {k: v.cpu() for k, v in global_state_gpu.items()}
    results = []
    total_r = 0

    for ci_, concept in enumerate(concepts):
        name = concept['name']
        if on_concept_start and ci_ >= 1:
            on_concept_start(ci_, global_state_cpu)

        # Slice each client's concept data into R_PER_CONCEPT equal slices
        client_slices = []
        for Xk, yk in concept['client_data']:
            if len(Xk) == 0:
                client_slices.append([(None, None)] * R_PER_CONCEPT)
                continue
            splits = np.array_split(np.arange(len(Xk)), R_PER_CONCEPT)
            client_slices.append([
                (Xk[sp], yk[sp]) if len(sp) > 0 else (None, None)
                for sp in splits
            ])

        for r_local in range(R_PER_CONCEPT):
            updates = []
            for ci, k_slices in enumerate(client_slices):
                Xsl, ysl = k_slices[r_local]
                if Xsl is None or len(Xsl) == 0:
                    continue
                X_gpu = torch.tensor(Xsl).float().to(device)
                y_gpu = torch.tensor(ysl).long().to(device)
                state_cpu, n = local_train_ids(
                    global_state_gpu, X_gpu, y_gpu, device, client_id=ci
                )
                updates.append((state_cpu, n))

            if updates:
                global_state_cpu = fedavg_aggregate(updates)
                global_state_gpu = {k: v.to(device)
                                    for k, v in global_state_cpu.items()}
            total_r += 1

        overall, per_class, macro_f1, per_recall = evaluate_ids(
            global_state_cpu, test_set[0], test_set[1], device
        )
        results.append({'event': name, 'round': total_r,
                        'overall': overall, 'macro_f1': macro_f1,
                        **_metrics(per_class, per_recall)})
        if verbose:
            print(f"  After {name:12s} (r={total_r:2d}) | "
                  f"Acc={overall:.4f} | MacroF1={macro_f1:.4f}")

    if verbose:
        print(f"  [Non-stationary] Final: Acc={results[-1]['overall']:.4f} "
              f"MacroF1={results[-1]['macro_f1']:.4f}")
    return global_state_cpu, results


# ── Experiment C: CDA-FedAvg IDS ──────────────────────────────────────────────

def run_cda_fedavg_ids(concepts, test_set, device, n_classes=N_CLASSES,
                       verbose=True, always_adapt=False,
                       detector_mode='conf', fusion='or', on_concept_start=None):
    """
    CDA-FedAvg on CICIDS2017 — the novel thesis contribution.

    Algorithm 4 (client) per concept:
      Phase 0 (Monday): initial training on BENIGN baseline data
      Phases 1-4: stream each day's data through detector,
                  run driftAdaptation if drift fires

    Algorithm 3 (server): aggregate after every training round.

    Key difference from HAR:
      - Concepts = CICIDS2017 days (temporal ordering)
      - Model = IDSDNN (tabular MLP, not CNN)
      - Non-IID clients via Dirichlet(α=0.5)
      - New attack classes appear per concept (not same classes throughout)
    """
    print("\n── Experiment C: CDA-FedAvg IDS ─────────────────────────")
    print(f"  λ={LAM}  Δ={DELTA}  Nmax={N_MAX}  L={L_CAPACITY}  "
          f"PER_CLASS={PER_CLASS}")

    # Per-client components
    memories  = [LongTermMemory(rng=np.random.default_rng([ids_loader.SEED, 20_000 + k]))
                 for k in range(N_CLIENTS)]
    # per-client RNG: the stochastic check no longer draws from the global
    # NumPy stream, so clients cannot couple and results cannot depend on what
    # ran before them in the same process.
    # detector_mode='conf' delegates every call to a single confidence channel,
    # so it is bit-identical to the plain BetaDriftDetector.
    detectors = [HybridDriftDetector(
                     BetaDriftDetector, mode=detector_mode, fusion=fusion,
                     rng=np.random.default_rng([ids_loader.SEED, k]),
                     rng2=np.random.default_rng([ids_loader.SEED, 10_000 + k]))
                 for k in range(N_CLIENTS)]
    aes = [None] * N_CLIENTS
    scales = [1.0] * N_CLIENTS

    global_state_gpu = IDSDNN(n_classes=n_classes).to(device).state_dict()
    global_state_cpu = {k: v.cpu() for k, v in global_state_gpu.items()}

    results          = []
    total_rounds     = 0
    detected_days    = []

    # ── Phase 0: Initial Monday (BENIGN baseline) training ───────────────────
    # Mirrors HAR Algorithm 4 line 3: L ← driftAdaptation(L, L, B, R, E, η)
    monday = concepts[0]
    if verbose:
        print(f"\n  [P0] {monday['name']}: collect BENIGN → {R_PER_CONCEPT} rounds")

    # Collect Monday data into memory
    for k, (Xk, yk) in enumerate(monday['client_data']):
        if len(Xk) > 0:
            memories[k].collect_and_expand(Xk, yk, monday['name'])

    # Train R rounds on Monday memory
    for r in range(1, R_PER_CONCEPT + 1):
        updates = []
        for k in range(N_CLIENTS):
            X_L, y_L = memories[k].get_all()
            if X_L is None or len(X_L) < 2:
                continue
            state_cpu, n = local_train_ids(
                global_state_gpu,
                torch.tensor(X_L).float().to(device),
                torch.tensor(y_L).long().to(device),
                device, client_id=k
            )
            updates.append((state_cpu, n))
        if updates:
            global_state_cpu = fedavg_aggregate(updates)
            global_state_gpu = {k: v.to(device) for k, v in global_state_cpu.items()}
        total_rounds += 1

    overall, per_class, macro_f1, per_recall = evaluate_ids(
        global_state_cpu, test_set[0], test_set[1], device
    )
    results.append({'event': 'Monday_init', 'round': total_rounds,
                    'overall': overall, 'macro_f1': macro_f1,
                    'memory_rows': [m.total_samples() for m in memories],
                    **_metrics(per_class, per_recall)})
    if verbose:
        print(f"    [Monday trained ] r={total_rounds:2d} | "
              f"Acc={overall:.4f} | MacroF1={macro_f1:.4f}")

    # Warm up detector windows with Monday confidence scores
    # (detector needs high-confidence BENIGN context before streaming new days)
    warm_model = IDSDNN(n_classes=n_classes).to(device)
    warm_model.load_state_dict(global_state_gpu)
    warm_model.eval()

    # Reconstruction channel: one small AE per client, fitted on that client's
    # phase-0 traffic only. Local by design — drift detection is a per-client
    # decision and each client's notion of "normal" is its own.
    if detector_mode in ('recon', 'hybrid'):
        for k, (Xk, yk) in enumerate(monday['client_data']):
            if len(Xk) < 2:
                continue
            aes[k], scales[k] = train_ae(Xk, device, N_FEATURES,
                                         seed=ids_loader.SEED * 100 + k)
        if verbose:
            print(f"    [AE] trained on phase-0, median-error scales "
                  f"{min(s for s in scales):.3g}–{max(scales):.3g}")

    for k, (Xk, yk) in enumerate(monday['client_data']):
        if len(Xk) == 0:
            continue
        X_t = torch.tensor(Xk).float().to(device)
        with torch.no_grad():
            confs = warm_model.get_confidence(X_t)
        rec = (recon_scores(aes[k], Xk, scales[k], device)
               if aes[k] is not None else None)
        for i, conf in enumerate(confs):
            detectors[k].update(float(conf),
                                float(rec[i]) if rec is not None else None)

    # ── Phases 1-4: Stream Tuesday → Friday ──────────────────────────────────
    for ci_, concept in enumerate(concepts[1:], start=1):
        name = concept['name']
        desc = concept['description']
        if on_concept_start:
            on_concept_start(ci_, global_state_cpu)

        # Build stream model for confidence estimation
        stream_model = IDSDNN(n_classes=n_classes).to(device)
        stream_model.load_state_dict(global_state_gpu)
        stream_model.eval()

        # Stream each client's data through its detector (Algorithm 4, lines 5-13)
        clients_detected = 0
        channels = []
        fired_n, total_n = 0, sum(len(Xk) for Xk, _ in concept['client_data'])          # which channel fired, per client, this concept

        if always_adapt:
            # Ablation: rehearsal with a perfect oracle detector. Adapting at
            # every concept boundary isolates what the Beta-CUSUM detector
            # contributes on top of the rehearsal buffer itself.
            clients_detected = N_CLIENTS
        else:
            for k, (Xk, yk) in enumerate(concept['client_data']):
                if len(Xk) == 0:
                    continue
                X_t = torch.tensor(Xk).float().to(device)
                with torch.no_grad():
                    confs = stream_model.get_confidence(X_t)
                rec = (recon_scores(aes[k], Xk, scales[k], device)
                       if aes[k] is not None else None)

                for i, conf in enumerate(confs):
                    conf_f = float(conf)
                    rec_f = float(rec[i]) if rec is not None else None
                    detectors[k].update(conf_f, rec_f)

                    # Stochastic drift check: P(run) = e^{-2qi} (Alg. 4 line 12),
                    # now runnable from either channel using its own score
                    if detectors[k].should_check(conf_f, rec_f):
                        if detectors[k].detect():
                            channels.append(detectors[k].fired_by[-1])
                            detectors[k].reset()
                            clients_detected += 1
                            fired_n += len(Xk)
                            break

        # Algorithm 3: aggregate if any client detected drift
        fired_share = fired_n / max(total_n, 1)
        triggered = (always_adapt
                     or (clients_detected >= QUORUM if QUORUM_MODE == 'count'
                         else fired_share >= QUORUM_WEIGHT))
        if triggered:
            # record which concept fired and how many of the 10 clients agreed —
            # the per-client count is what the α-sweep analysis needs
            detected_days.append({
                'concept': name, 'clients': clients_detected,
                'channels': {c: channels.count(c)
                             for c in ('conf', 'recon', 'both') if channels.count(c)}})
            if verbose:
                print(f"\n  [DRIFT] {name:12s}: {clients_detected}/{N_CLIENTS} clients "
                      f"→ {R_PER_CONCEPT} adaptation rounds")
                print(f"          Attack: {desc}")

            # driftAdaptation for all clients (Algorithm 6)
            for r in range(1, R_PER_CONCEPT + 1):
                updates = []
                for k, (Xk, yk) in enumerate(concept['client_data']):
                    if r == 1 and len(Xk) > 0:  # collect + expand (lines 1-2)
                        memories[k].collect_and_expand(Xk, yk, name)
                    X_L, y_L = memories[k].get_all()
                    if X_L is None or len(X_L) < 2:
                        continue
                    state_cpu, n = local_train_ids(
                        global_state_gpu,
                        torch.tensor(X_L).float().to(device),
                        torch.tensor(y_L).long().to(device),
                        device, client_id=k
                    )
                    updates.append((state_cpu, n))

                if updates:
                    global_state_cpu = fedavg_aggregate(updates)
                    global_state_gpu = {k: v.to(device)
                                        for k, v in global_state_cpu.items()}
                total_rounds += 1

            overall, per_class, macro_f1, per_recall = evaluate_ids(
                global_state_cpu, test_set[0], test_set[1], device
            )
            results.append({'event': name, 'round': total_rounds,
                            'overall': overall, 'macro_f1': macro_f1,
                            'memory_rows': [m.total_samples() for m in memories],
                            **_metrics(per_class, per_recall)})
            detected_days[-1]['memory_client0'] = memories[0].total_samples()
            if verbose:
                print(f"    [After {name:8s}] r={total_rounds:2d} | "
                      f"Acc={overall:.4f} | MacroF1={macro_f1:.4f}")
                mem_str = f"{memories[0].total_samples()} samples " \
                          f"({memories[0].n_concepts()} concepts)"
                print(f"    Memory: {mem_str}")

        else:
            if verbose:
                print(f"\n  [NO DRIFT] {name}: 0/{N_CLIENTS} clients fired")

    if verbose:
        print(f"\n  [CDA-FedAvg Summary]")
        print(f"    Total rounds    : {total_rounds}")
        print(f"    Drifts detected : {detected_days}")

    return global_state_cpu, results, detected_days


# ── Print comparison table ────────────────────────────────────────────────────

def print_comparison(res_iid, res_ns, res_cda):
    """Print final results for all three experiments side by side."""
    print(f"\n{'='*65}")
    print("IDS CDA-FedAvg RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"{'Experiment':<28} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"{'─'*65}")

    def last(results):
        return results[-1]['overall'], results[-1]['macro_f1']

    acc_iid, f1_iid = last(res_iid)
    acc_ns,  f1_ns  = last(res_ns)
    acc_cda, f1_cda = last(res_cda)

    print(f"  IID FedAvg (upper bound) {acc_iid:>12.4f} {f1_iid:>10.4f}")
    print(f"  Non-stationary FedAvg    {acc_ns:>12.4f} {f1_ns:>10.4f}")
    print(f"  CDA-FedAvg (ours)        {acc_cda:>12.4f} {f1_cda:>10.4f}")
    print(f"{'─'*65}")
    print(f"  Recovery vs Non-stat:    "
          f"{acc_cda-acc_ns:>+11.4f} {f1_cda-f1_ns:>+9.4f}")

    print(f"\n── Per-class F1 (final CDA-FedAvg model) ──")
    final = res_cda[-1]
    print(f"  {'Class':<14} {'F1':>8}")
    for cls in CLASS_NAMES:
        val = final.get(f'f1_{cls}', float('nan'))
        flag = '  ← minority' if cls in ('WebAttack', 'Bot', 'BruteForce') else ''
        print(f"  {cls:<14} {val:>8.4f}{flag}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print("IDS CDA-FedAvg Extension — CICIDS2017")
    print(f"{'='*60}")
    print(f"Device  : {device}")
    print(f"Clients : {N_CLIENTS}  |  α={ALPHA}  |  Rounds={N_ROUNDS}")
    count_parameters_ids()

    # ── Load data ─────────────────────────────────────────────────────────────
    max_pc = 500 if QUICK_TEST else MAX_PER_CLASS
    print(f"\n{'QUICK_TEST' if QUICK_TEST else 'Full'}: max_per_class={max_pc}")

    concepts, test_set, concept_tests, scaler, feat_cols = load_ids_dataset(
        n_clients=N_CLIENTS,
        alpha=ALPHA,
        max_per_class=max_pc,
        test_fraction=TEST_FRACTION,
        verbose=True
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── Experiment A: IID FedAvg ──────────────────────────────────────────────
    t0 = time.time()
    _, res_iid = run_iid_fedavg(concepts, test_set, device)
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")

    # ── Experiment B: Non-stationary FedAvg ───────────────────────────────────
    t0 = time.time()
    _, res_ns = run_nonstationary_fedavg(concepts, test_set, device)
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")

    # ── Experiment C: CDA-FedAvg ──────────────────────────────────────────────
    t0 = time.time()
    cda_state, res_cda, detected = run_cda_fedavg_ids(concepts, test_set, device)
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")

    # ── Print results ─────────────────────────────────────────────────────────
    print_comparison(res_iid, res_ns, res_cda)

    # ── Per-class F1 deep dive (minority attack classes) ─────────────────────
    print(f"\n── Catastrophic forgetting on minority classes (Non-stat vs CDA) ──")
    for cls in ['WebAttack', 'Bot', 'BruteForce', 'DoS', 'DDoS', 'PortScan']:
        ns_f1  = res_ns[-1].get(f'f1_{cls}', float('nan'))
        cda_f1 = res_cda[-1].get(f'f1_{cls}', float('nan'))
        delta  = cda_f1 - ns_f1 if not (np.isnan(ns_f1) or np.isnan(cda_f1)) else float('nan')
        print(f"  {cls:<14} Non-stat={ns_f1:.4f}  CDA={cda_f1:.4f}  Δ={delta:+.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    save_path = os.path.join(CKPT_DIR, 'ids_results.json')
    with open(save_path, 'w') as f:
        json.dump({
            'iid':         res_iid,
            'nonstat':     res_ns,
            'cda':         res_cda,
            'detected':    detected,
            'n_clients':   N_CLIENTS,
            'alpha':       ALPHA,
            'quick_test':  QUICK_TEST,
        }, f, indent=2, default=lambda x: None if np.isnan(x) else x)
    print(f"\n[SAVED] {save_path}")


if __name__ == '__main__':
    main()
