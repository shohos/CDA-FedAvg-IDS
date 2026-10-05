"""
aggregators.py -- server aggregation rules for the aggregator study (2026-09-27).

  'fedavg'   weighted parameter average (McMahan et al., 2017). The pipelines'
             own code path; this module is not involved, so results are unchanged.
  'fedprox'  FedAvg plus a proximal term (mu/2)||w - w_global||^2 in every
             client's local loss (Li et al., MLSys 2020).
  'scaffold' SCAFFOLD (Karimireddy et al., ICML 2020). Each local gradient step is
             corrected by (c - c_i). The client variate c_i is refreshed by
             option I of the paper: the gradient of the client's local loss at the
             server model, which does not depend on the local optimiser (ours uses
             momentum). Both model and control averages use the declared active
             client sample weights. Controls are reweighted before each round.
             The option-I pass preserves RNG and model buffers. This weighted
             full-participation adaptation is not the original uniform objective.
  'median'   coordinate-wise median of the client models (Yin et al., ICML 2018),
             unweighted; with an even number of clients, the mean of the two middle
             values.

Client variates persist across rounds and concepts for a client index; reset()
clears all state and must be called at the start of every run.
"""
import torch

AGG = 'fedavg'   # 'fedavg' | 'fedprox' | 'scaffold' | 'median'
MU = 0.01        # FedProx proximal weight (chosen on development seeds)
N_CLIENTS = 10
_S = {'c': None, 'ck': {}, 'pending': [], 'weights': None, 'seen': set(), 'gradient_rows': 0}


def reset():
    _S.update(c=None, ck={}, pending=[], weights=None, seen=set(), gradient_rows=0)


def begin_round(sizes):
    """Declare active clients and sample weights BEFORE local updates.

    Reweight stored controls to the round's current sample-weighted objective.
    Zero-initialize unseen clients; inactive controls persist but have zero
    weight for this round. This is a full-active-participation weighted
    SCAFFOLD-derived extension, not the source paper's sampling theorem.
    """
    if AGG != 'scaffold':
        return
    if _S['pending']:
        raise RuntimeError('previous SCAFFOLD round was not aggregated')
    sizes = {int(k): int(n) for k, n in sizes.items() if n >= 2}
    if not sizes:
        _S['weights'] = {}
        return
    total = sum(sizes.values())
    _S['weights'] = {k: n / total for k, n in sizes.items()}
    _S['seen'] = set()
    template = next(iter(_S['ck'].values()), None)
    _S['c'] = None if template is None else [
        sum((_S['ck'][k][j] * w for k, w in _S['weights'].items()
             if k in _S['ck']), torch.zeros_like(t))
        for j, t in enumerate(template)]



class LocalHook:
    """Client-side part of FedProx / SCAFFOLD for one local training call."""

    def __init__(self, model, client_id, X, y, crit, B):
        self.params = list(model.parameters())
        if AGG == 'fedprox':
            self.anchor = [p.detach().clone() for p in self.params]
        if AGG == 'scaffold':
            assert client_id is not None, 'SCAFFOLD needs a client id'
            if _S['weights'] is None or client_id not in _S['weights']:
                raise RuntimeError('declare active client sample sizes with begin_round first')
            if client_id in _S['seen']:
                raise RuntimeError('duplicate client in SCAFFOLD round')
            _S['seen'].add(client_id)
            zeros = [torch.zeros_like(p) for p in self.params]
            c = _S['c'] or zeros
            self.cid, self.ci_old = client_id, _S['ck'].get(client_id) or zeros
            self.corr = [a - b for a, b in zip(c, self.ci_old)]
            self.ci_new = local_gradient(model, X, y, crit, B)

    def penalty(self):
        if AGG != 'fedprox':
            return 0.0
        return 0.5 * MU * sum(((p - a) ** 2).sum() for p, a in zip(self.params, self.anchor))

    def correct(self):
        if AGG == 'scaffold':
            for p, d in zip(self.params, self.corr):
                if p.grad is not None:
                    p.grad.add_(d)

    def finish(self):
        if AGG == 'scaffold':
            _S['pending'].append(self.cid)
            _S['ck'][self.cid] = self.ci_new


def local_hook(model, client_id, X, y, crit, B):
    """None for FedAvg and Median, so their local training is untouched."""
    return LocalHook(model, client_id, X, y, crit, B) if AGG in ('fedprox', 'scaffold') else None


def local_gradient(model, X, y, crit, B):
    """Mean gradient of the local loss at the current (server) weights."""
    grads = [torch.zeros_like(p) for p in model.parameters()]
    n = len(X)
    # Retain training-mode dropout, but isolate its draws and all model buffers.
    # Match the pipeline's skip-singleton objective and normalize by rows used.
    batches = [(s, min(B, n-s)) for s in range(0, n, B) if min(B, n-s) >= 2]
    used = sum(m for _, m in batches)
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            for s, m in batches:
                model.zero_grad()
                (crit(model(X[s:s+m]), y[s:s+m]) * (m / used)).backward()
                for g, p in zip(grads, model.parameters()):
                    if p.grad is not None:
                        g.add_(p.grad.detach())
    finally:
        with torch.no_grad():
            for name, b in model.named_buffers():
                b.copy_(buffers[name])
        model.zero_grad()
    _S['gradient_rows'] += used
    return grads


def aggregate(updates, weighted_mean):
    """updates: [(state_dict_cpu, n_samples)]. weighted_mean: the pipeline's FedAvg."""
    if AGG == 'median':
        return {k: torch.quantile(torch.stack([s[k].float() for s, _ in updates]), 0.5, dim=0)
                for k in updates[0][0]}
    if AGG == 'scaffold':
        active = [(state, n) for state, n in updates if n >= 2]
        if not active:
            _S['weights'] = None
            return updates[0][0]
        updates = active
    out = weighted_mean(updates)
    if AGG == 'scaffold' and _S['pending']:
        if set(_S['pending']) != set(_S['weights']):
            raise RuntimeError('SCAFFOLD active client set does not match the declared round')
        if len(updates) != len(_S['pending']):
            raise RuntimeError('model and control update counts differ')
        total = sum(n for _, n in updates)
        for cid, (_, n) in zip(_S['pending'], updates):
            if abs(n / total - _S['weights'][cid]) > 1e-12:
                raise RuntimeError('model/control sample weights disagree')
        template = _S['ck'][_S['pending'][0]]
        _S['c'] = [sum((_S['ck'][k][j] * w for k, w in _S['weights'].items()),
                      torch.zeros_like(t)) for j, t in enumerate(template)]
        _S['pending'] = []
        _S['weights'] = None
    return out


