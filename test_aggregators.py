"""Median and FedProx checks (moved here from the old aggregators.py self-test).

    python test_aggregators.py
"""
import torch
import torch.nn as nn
import aggregators as A

# median: element-wise middle of three models, sample counts ignored
ups = [({'w': torch.tensor([1., 5.])}, 10), ({'w': torch.tensor([3., 1.])}, 1),
       ({'w': torch.tensor([2., 9.])}, 1)]
A.AGG = 'median'
assert torch.equal(A.aggregate(ups, None)['w'], torch.tensor([2., 5.]))
# even count: mean of the two middle values
assert torch.allclose(A.aggregate(ups + [({'w': torch.tensor([4., 0.])}, 1)], None)['w'],
                      torch.tensor([2.5, 3.0]))

# fedprox: no penalty at the anchor, positive after a move
torch.manual_seed(0)
m = nn.Linear(3, 2)
A.AGG = 'fedprox'
h = A.local_hook(m, 0, None, None, None, 4)
assert float(h.penalty()) == 0.0
with torch.no_grad():
    m.weight.add_(1.0)
assert float(h.penalty()) > 0

# fedavg and median add no client-side hook
for rule in ('fedavg', 'median'):
    A.AGG = rule
    assert A.local_hook(m, 0, None, None, None, 4) is None
print('aggregators ok')
