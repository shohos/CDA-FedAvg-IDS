"""
tabfm_frontend.py — a REAL front-end-only TabFM adapter, and the measurements
that were missing.

Why this exists. `tabfm_embed.py` extracts row representations with a forward
hook on `row_interactor_2`, which is fine for checking that the representation
exists but does NOT avoid any work: it still calls `clf.predict(Xq)`, so the
1.62-billion-parameter `icl_predictor` runs anyway. Every latency number the
project has quoted for "the front end" is therefore full-model cost. The mentor
review of 2026-09-18 (C2) called this out and it is correct.

The model's own forward pass is:

    x    = nan_to_num(x)
    emb  = cell_embedder(x, y, train_size, cat_mask, d)
    emb  = col_embedder(emb, train_size)
    emb  = cat([cls_tokens, emb], dim=2)
    emb  = row_interactor(emb, d)
    emb  = col_embedder_2(emb, train_size)
    reps = row_interactor_2(emb, d)          <-- the representation. STOP HERE.
    return icl_predictor(reps, y, train_size)   <-- 1.62B params, skipped

HOW THIS SKIPS IT WITHOUT REIMPLEMENTING PREPROCESSING. The wrapper's
preprocessing is substantial — ordinal encoding, outlier removal, quantile/
standard normalisation, feature shuffling, padding, categorical masks and the
`d` sequence lengths — and reimplementing it would be a second source of truth
and a bug farm. So instead of rebuilding the pipeline, this adapter temporarily
substitutes `model.icl_predictor` with a capture stub that records `reps` and
returns a correctly shaped zero logit tensor. The wrapper's own code path runs
end to end, the preprocessing is exact by construction, and the expensive stage
never executes.

The stub returns zeros, so any prediction the wrapper computes from it is
meaningless and is discarded. Only `reps` is used.

WHAT IS VERIFIED (`--verify`), because an adapter that is subtly wrong is worse
than no adapter:

  equivalence    front-end reps == hook reps, within a declared tolerance
  row order      permuting the query permutes the embeddings the same way
  batching       one call and several chunked calls agree
  determinism    the same inputs twice give bit-identical output
  context bind   changing the reference context changes the embedding
                 (it must, since the map is transductive -- a version guard
                 depends on this being true)

WHAT IS MEASURED (`--bench`), preprocessing included, as the review asked:

  cold setup     first call, including any lazy init
  warm batch     throughput on a full batch, per row
  batch-1        single-flow latency, mean and p95 -- the deployable pattern
  peak memory    torch.cuda.max_memory_allocated around each path
  full model     the same measurements WITH the ICL stage, for the ratio

Out: checkpoints/tabfm_frontend.json
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ids_loader, unsw_loader
from ids_loader import load_ids_dataset, CKPT_DIR
from unsw_loader import load_unsw_dataset

TOL = 1e-4          # declared equivalence tolerance, float32 attention stack


class _CaptureHead(nn.Module):
    """Stands in for `icl_predictor`. Records `reps` and returns a zero logit
    tensor of the shape the real head would, so the wrapper's downstream code
    runs unchanged. The zeros are discarded."""

    def __init__(self, max_classes):
        super().__init__()
        self.max_classes = max_classes
        self.buf = []

    def forward(self, reps, y, train_size):
        self.buf.append(reps.detach())
        b, t, _ = reps.shape
        return torch.zeros(b, t, self.max_classes,
                           dtype=reps.dtype, device=reps.device)


class FrontEnd:
    """Frozen TabFM front end behind a fixed reference context.

    The map is transductive: a row's embedding depends on the context it is
    embedded alongside, so `fit()` pins the reference context and every later
    `embed()` uses that same one. `context_id` changes whenever the context
    does, and is meant to be stored beside any centroid computed from these
    coordinates -- centroids from different context versions are not comparable.
    """

    def __init__(self, model, device, n_estimators=1):
        from tabfm import TabFMClassifier
        self.model, self.device, self._C = model, device, TabFMClassifier
        self.n_estimators = n_estimators        # >1 reshuffles features per
        self.clf = None                         # estimator: not a function
        self.context_id = None

    def fit(self, Xctx, yctx):
        self.clf = self._C(model=self.model, n_estimators=self.n_estimators,
                           random_state=42)
        self.clf.fit(Xctx, yctx)
        self.n_ctx = len(Xctx)
        h = __import__('hashlib').sha256()
        h.update(np.ascontiguousarray(Xctx, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(yctx).tobytes())
        self.context_id = h.hexdigest()[:16]
        return self

    def _run(self, Xq, full_model):
        """One pass over Xq. `full_model=False` swaps in the capture stub."""
        if full_model:
            self.clf.predict(Xq)
            return None
        real = self.model.icl_predictor
        stub = _CaptureHead(getattr(self.model, 'max_classes', 10)).to(self.device)
        self.model.icl_predictor = stub
        try:
            self.clf.predict(Xq)
        finally:
            self.model.icl_predictor = real     # always restore
        r = torch.cat([b.reshape(-1, b.shape[-1]) for b in stub.buf], 0)
        # the wrapper feeds context+query together; keep the trailing query rows
        return r[-len(Xq):].float().cpu().numpy()

    def embed(self, Xq):
        if self.clf is None:
            raise RuntimeError('call fit() with a reference context first')
        return self._run(np.asarray(Xq), full_model=False)


def hook_embed(model, clf, Xq, device):
    """The OLD path: full model + forward hook. Used only to verify the new one."""
    buf = []
    h = model.row_interactor_2.register_forward_hook(
        lambda m, i, o: buf.append(o.detach()))
    try:
        clf.predict(Xq)
    finally:
        h.remove()
    r = torch.cat([b.reshape(-1, b.shape[-1]) for b in buf], 0)
    return r[-len(Xq):].float().cpu().numpy()


def balanced(X, y, n, rng):
    cls = np.unique(y)
    per = max(1, n // len(cls))
    idx = np.concatenate([rng.choice(np.where(y == c)[0],
                                     min((y == c).sum(), per), replace=False)
                          for c in cls])
    return idx[rng.permutation(len(idx))]


def peak_mb(device):
    return (torch.cuda.max_memory_allocated(device) / 2**20
            if device == 'cuda' else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['cicids', 'unsw'], default='cicids')
    ap.add_argument('--context', type=int, default=600)
    ap.add_argument('--verify-n', type=int, default=64)
    ap.add_argument('--bench-n', type=int, default=256)
    ap.add_argument('--single-reps', type=int, default=20)
    ap.add_argument('--out', default='tabfm_frontend.json')
    a = ap.parse_args()

    from tabfm.src.pytorch import tabfm_v1_0_0 as M
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = M.load('classification', device=dev)
    front_params = sum(sum(p.numel() for p in getattr(model, n).parameters())
                       for n in ('cell_embedder', 'col_embedder', 'col_embedder_2',
                                 'row_interactor', 'row_interactor_2'))
    icl_params = sum(p.numel() for p in model.icl_predictor.parameters())
    print(f'front end {front_params:,} params | icl {icl_params:,} params | {dev}',
          flush=True)

    ids_loader.SEED = unsw_loader.SEED = 42
    load = load_ids_dataset if a.dataset == 'cicids' else load_unsw_dataset
    concepts, (Xte, yte), *_ = load(n_clients=10, alpha=0.5, max_per_class=5000,
                                    verbose=False)
    X = np.concatenate([xk for c in concepts for xk, _ in c['client_data'] if len(xk)])
    y = np.concatenate([yk for c in concepts for _, yk in c['client_data'] if len(yk)])
    rng = np.random.default_rng(0)
    ctx = balanced(X, y, a.context, rng)
    Xctx, yctx = X[ctx], y[ctx]

    fe = FrontEnd(model, dev).fit(Xctx, yctx)
    print(f'reference context {len(Xctx)} rows, context_id {fe.context_id}\n',
          flush=True)
    res = {'meta': {'dataset': a.dataset, 'device': dev, 'context': len(Xctx),
                    'context_id': fe.context_id, 'tolerance': TOL,
                    'front_end_params': int(front_params),
                    'icl_params': int(icl_params)}}

    # ---------------- verification ---------------------------------------
    q = rng.choice(len(Xte), a.verify_n, replace=False)
    Xq = Xte[q]
    v = {}

    E_new = fe.embed(Xq)
    E_hook = hook_embed(model, fe.clf, Xq, dev)
    d = float(np.abs(E_new - E_hook).max())
    v['equivalence_max_abs_diff'] = d
    v['equivalence_ok'] = bool(d <= TOL)
    v['embed_dim'] = int(E_new.shape[1])
    print(f'[equivalence]  max|front-end - hook| = {d:.3e}   '
          f'{"OK" if d <= TOL else "FAIL"}  (dim {E_new.shape[1]})', flush=True)

    perm = rng.permutation(len(Xq))
    E_perm = fe.embed(Xq[perm])
    d = float(np.abs(E_perm - E_new[perm]).max())
    v['row_order_max_abs_diff'] = d
    v['row_order_ok'] = bool(d <= TOL)
    print(f'[row order]    max|embed(Xq[p]) - embed(Xq)[p]| = {d:.3e}   '
          f'{"OK" if d <= TOL else "FAIL"}', flush=True)

    half = len(Xq) // 2
    E_chunk = np.concatenate([fe.embed(Xq[:half]), fe.embed(Xq[half:])])
    d = float(np.abs(E_chunk - E_new).max())
    v['batching_max_abs_diff'] = d
    v['batching_ok'] = bool(d <= TOL)
    print(f'[batching]     max|chunked - single call| = {d:.3e}   '
          f'{"OK" if d <= TOL else "FAIL"}'
          f'{"   <- transductive: chunks see different neighbours" if d > TOL else ""}',
          flush=True)

    d = float(np.abs(fe.embed(Xq) - E_new).max())
    v['determinism_max_abs_diff'] = d
    v['determinism_ok'] = bool(d == 0.0)
    print(f'[determinism]  max|repeat - first| = {d:.3e}   '
          f'{"OK" if d == 0.0 else "FAIL"}', flush=True)

    ctx2 = balanced(X, y, a.context, np.random.default_rng(7))
    fe2 = FrontEnd(model, dev).fit(X[ctx2], y[ctx2])
    d = float(np.abs(fe2.embed(Xq) - E_new).max())
    v['context_sensitivity_max_abs_diff'] = d
    v['context_changes_the_map'] = bool(d > TOL)
    v['context_id_differs'] = bool(fe2.context_id != fe.context_id)
    print(f'[context bind] max|other context - this one| = {d:.3e}   '
          f'{"map moves, as expected" if d > TOL else "FAIL: map did not move"}',
          flush=True)
    res['verification'] = v

    # ---------------- benchmark ------------------------------------------
    Xb = Xte[rng.choice(len(Xte), a.bench_n, replace=False)]
    b = {'bench_n': int(len(Xb)), 'single_reps': a.single_reps}

    def timed(fn, sync=True):
        if dev == 'cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        fn()
        if sync and dev == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter() - t0, peak_mb(dev)

    b['cold_setup_s'], b['cold_peak_mb'] = timed(lambda: fe.embed(Xb[:8]))
    t, m = timed(lambda: fe.embed(Xb))
    b['front_batch_s'], b['front_batch_peak_mb'] = t, m
    b['front_ms_per_row'] = 1000 * t / len(Xb)
    t, m = timed(lambda: fe._run(Xb, full_model=True))
    b['full_batch_s'], b['full_batch_peak_mb'] = t, m
    b['full_ms_per_row'] = 1000 * t / len(Xb)
    b['batch_speedup'] = b['full_batch_s'] / b['front_batch_s']

    singles_f, singles_u = [], []
    for i in range(a.single_reps):
        one = Xb[i:i + 1]
        singles_f.append(timed(lambda: fe.embed(one))[0] * 1000)
        singles_u.append(timed(lambda: fe._run(one, full_model=True))[0] * 1000)
    for tag, arr in (('front', singles_f), ('full', singles_u)):
        b[f'{tag}_single_ms_mean'] = float(np.mean(arr))
        b[f'{tag}_single_ms_p95'] = float(np.percentile(arr, 95))
    b['single_speedup'] = b['full_single_ms_mean'] / b['front_single_ms_mean']
    res['benchmark'] = b

    print(f"\n[batch {len(Xb)}]  front end {b['front_ms_per_row']:.2f} ms/row "
          f"(peak {b['front_batch_peak_mb']:.0f} MB)  |  full model "
          f"{b['full_ms_per_row']:.2f} ms/row (peak {b['full_batch_peak_mb']:.0f} MB)"
          f"  |  {b['batch_speedup']:.2f}x")
    print(f"[batch 1]     front end {b['front_single_ms_mean']:.1f} ms mean / "
          f"{b['front_single_ms_p95']:.1f} ms p95  |  full model "
          f"{b['full_single_ms_mean']:.1f} / {b['full_single_ms_p95']:.1f}"
          f"  |  {b['single_speedup']:.2f}x")

    out = os.path.join(CKPT_DIR, a.out)
    if os.path.exists(out):
        sys.exit(f'{out} exists; choose another --out')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'\n[SAVED] checkpoints/{a.out}')

    bad = [k for k, val in v.items() if k.endswith('_ok') and not val]
    if bad:
        print(f'!! failed checks: {bad}')
        sys.exit(1)


if __name__ == '__main__':
    main()
