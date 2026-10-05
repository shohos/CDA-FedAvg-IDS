import sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch import nn
import aggregators as A
class Scalar(nn.Module):
 def __init__(self):super().__init__();self.w=nn.Parameter(torch.zeros(1,1))
 def forward(self,x):return x@self.w
def weighted(updates):return {k:sum(s[k]*n for s,n in updates)/sum(n for _,n in updates) for k in updates[0][0]}
A.AGG='scaffold';A.reset();xs=[torch.ones(9,1),torch.ones(3,1)];ys=[torch.zeros(9,1),torch.ones(3,1)*4]
A.begin_round({0:9,1:3})
for k in range(2):A.local_hook(Scalar(),k,xs[k],ys[k],nn.MSELoss(),10).finish()
A.aggregate([({'w':torch.zeros(1,1)},9),({'w':torch.zeros(1,1)},3)],weighted)
assert A._S['c'][0].item()==-2
A.begin_round({0:9,1:3});corrected=[]
for k in range(2):
 m=Scalar();h=A.local_hook(m,k,xs[k],ys[k],nn.MSELoss(),10);nn.MSELoss()(m(xs[k]),ys[k]).backward();h.correct();corrected.append(m.w.grad.item());h.finish()
assert corrected==[-2,-2]
A.aggregate([({'w':torch.zeros(1,1)},9),({'w':torch.zeros(1,1)},3)],weighted)
A.begin_round({0:3,1:9});assert A._S['c'][0].item()==-6
A.begin_round({1:9});assert A._S['c'][0].item()==-8
checks={'weighted_objective':corrected,'changing_weights':-6,'partial_active_set':-8}
for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
 torch.manual_seed(7);m=nn.Sequential(nn.Linear(4,12),nn.BatchNorm1d(12),nn.ReLU(),nn.Dropout(.3),nn.Linear(12,2)).to(device).train()
 x=torch.randn(20,4,device=device);y=torch.randint(0,2,(20,),device=device);states=[]
 for rule in ['fedavg','scaffold']:
  A.AGG=rule;A.reset();A.begin_round({0:20});z=copy.deepcopy(m);torch.manual_seed(123)
  rng=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone() if device=='cuda' else None
  buffers={k:b.clone() for k,b in z.named_buffers()};h=A.local_hook(z,0,x,y,nn.CrossEntropyLoss(),5)
  assert torch.equal(rng,torch.get_rng_state())
  if cuda is not None:assert torch.equal(cuda,torch.cuda.get_rng_state())
  assert all(torch.equal(b,buffers[k]) for k,b in z.named_buffers())
  o=torch.optim.SGD(z.parameters(),lr=.01,momentum=.9);o.zero_grad();nn.CrossEntropyLoss()(z(x),y).backward()
  if h:h.correct()
  o.step();states.append(torch.cat([v.detach().flatten() for v in z.parameters()]))
 assert torch.equal(*states);checks[device+'_zero_correction_exact']=True
print(json.dumps(checks,indent=2))
if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(checks,indent=2))  # optional output file
