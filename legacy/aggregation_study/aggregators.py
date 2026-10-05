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
             momentum). Server: c <- c + (1/N) sum_i (c_i_new - c_i_old). Models are
             averaged by sample weight, as in our FedAvg, so that the control
             variates are the only change (the paper averages without weights).
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
_S = {'c': None, 'ck': {}, 'pending': []}


def reset():
    _S['c'], _S['ck'], _S['pending'] = None, {}, []


class LocalHook:
    """Client-side part of FedProx / SCAFFOLD for one local training call."""

    def __init__(self, model, client_id, X, y, crit, B):
        self.params = list(model.parameters())
        if AGG == 'fedprox':
            self.anchor = [p.detach().clone() for p in self.params]
        if AGG == 'scaffold':
            assert client_id is not None, 'SCAFFOLD needs a client id'
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
            _S['pending'].append([a - b for a, b in zip(self.ci_new, self.ci_old)])
            _S['ck'][self.cid] = self.ci_new


def local_hook(model, client_id, X, y, crit, B):
    """None for FedAvg and Median, so their local training is untouched."""
    return LocalHook(model, client_id, X, y, crit, B) if AGG in ('fedprox', 'scaffold') else None


def local_gradient(model, X, y, crit, B):
    """Mean gradient of the local loss at the current (server) weights."""
    grads = [torch.zeros_like(p) for p in model.parameters()]
    n = len(X)
    for s in range(0, n, B):
        Xb, yb = X[s:s + B], y[s:s + B]
        if len(Xb) < 2:
            continue
        model.zero_grad()
        (crit(model(Xb), yb) * (len(Xb) / n)).backward()
        for g, p in zip(grads, model.parameters()):
            if p.grad is not None:
                g.add_(p.grad.detach())
    model.zero_grad()
    return grads


def aggregate(updates, weighted_mean):
    """updates: [(state_dict_cpu, n_samples)]. weighted_mean: the pipeline's FedAvg."""
    if AGG == 'median':
        return {k: torch.quantile(torch.stack([s[k].float() for s, _ in updates]), 0.5, dim=0)
                for k in updates[0][0]}
    out = weighted_mean(updates)
    if AGG == 'scaffold' and _S['pending']:
        dev = _S['pending'][0][0].device
        c = _S['c'] or [torch.zeros_like(t) for t in _S['pending'][0]]
        _S['c'] = [ci + sum(d[i] for d in _S['pending']).to(dev) / N_CLIENTS
                   for i, ci in enumerate(c)]
        _S['pending'] = []
    return out


if __name__ == '__main__':
    import torch.nn as nn
    torch.manual_seed(0)
    # median: element-wise middle of three models
    ups = [({'w': torch.tensor([1., 5.])}, 10), ({'w': torch.tensor([3., 1.])}, 1),
           ({'w': torch.tensor([2., 9.])}, 1)]
    AGG = 'median'
    assert torch.equal(aggregate(ups, None)['w'], torch.tensor([2., 5.]))
    # even count: mean of the two middle values
    assert torch.allclose(aggregate(ups + [({'w': torch.tensor([4., 0.])}, 1)], None)['w'],
                          torch.tensor([2.5, 3.0]))
    # fedprox: no penalty at the anchor, positive after a move
    m = nn.Linear(3, 2)
    AGG = 'fedprox'
    h = local_hook(m, 0, None, None, None, 4)
    assert float(h.penalty()) == 0.0
    with torch.no_grad():
        m.weight.add_(1.0)
    assert float(h.penalty()) > 0
    # scaffold: first round has zero correction; server variate moves by mean(dc)/N
    AGG = 'scaffold'
    reset()
    X, y = torch.randn(20, 3), torch.randint(0, 2, (20,))
    crit = nn.CrossEntropyLoss()
    h = local_hook(m, 7, X, y, crit, 5)
    assert all(float(c.abs().sum()) == 0 for c in h.corr)
    h.finish()
    aggregate([({'w': torch.zeros(1)}, 1)], lambda u: u[0][0])
    g = local_gradient(m, X, y, crit, 5)
    assert all(torch.allclose(c, gi / N_CLIENTS) for c, gi in zip(_S['c'], g))
    h2 = local_hook(m, 7, X, y, crit, 5)          # now c_i = g, c = g/N: correction = c - c_i
    assert all(torch.allclose(d, gi / N_CLIENTS - gi) for d, gi in zip(h2.corr, g))
    print('aggregators ok')
