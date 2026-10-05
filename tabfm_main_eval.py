"""Frozen TabFM representation inside the repaired main-stream CDA-FedAvg loop.
New exploratory study. See revision_2026-09-28_main_streams protocol.
"""
import argparse,copy,hashlib,json,time,sys,os,gc
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import final_eval as F
from tabfm_frontend import FrontEnd,_CaptureHead

def digest(*arrays):
 h=hashlib.sha256()
 for a in arrays:
  a=np.ascontiguousarray(a);h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
 return h.hexdigest()
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2),encoding='utf8');t.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['cicids','unsw','ids2018'],required=True)
 ap.add_argument('--seeds',default='42');ap.add_argument('--alpha',type=float,default=.5);ap.add_argument('--max-per-class',type=int,default=5000)
 ap.add_argument('--out',required=True);ap.add_argument('--context',type=int,default=64);ap.add_argument('--batch',type=int,default=256)
 a=ap.parse_args();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise RuntimeError('refusing to overwrite an existing result')
 torch.set_num_threads(4);F.set_impl('repaired','raw');F.aggregators.AGG='fedavg'
 from tabfm.src.pytorch import tabfm_v1_0_0 as M
 t=time.perf_counter()
 from safetensors import safe_open
 checkpoint=Path.home()/'.cache/huggingface/hub/models--google--tabfm-1.0.0-pytorch/snapshots/77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83/classification'
 config=json.loads((checkpoint/'config.json').read_text())
 with torch.device('meta'):
  model=M.TabFM_HF(**config)
 full_params=sum(p.numel() for p in model.parameters())
 model.icl_predictor=_CaptureHead(model.max_classes)
 needed=set(model.state_dict());weights={};wh=hashlib.sha256()
 with safe_open(str(checkpoint/'model.safetensors'),framework='pt',device='cpu') as sf:
  for k in sorted(needed):
   v=sf.get_tensor(k);wh.update(k.encode());wh.update(v.numpy().tobytes());weights[k]=v
 model.load_state_dict(weights,strict=True,assign=True)
 model=model.to(dtype=torch.float32,device='cuda').eval()
 del weights;gc.collect();torch.cuda.empty_cache()
 checkpoint_front_sha256=wh.hexdigest()
 front_params=sum(p.numel() for p in model.parameters());front_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
 root=Path(__file__).parent
 files=F.SOURCES+['tabfm_main_eval.py','tabfm_frontend.py']
 result={'meta':vars(a)|{'status':'development','source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},'torch':torch.__version__,'full_params':full_params,'front_params':front_params,'front_bytes':front_bytes,'load_seconds':time.perf_counter()-t,'projection_seed':20260928,'context_policy':'largest initial client, benign only, shared once','icl_removed_before_cuda':True,'checkpoint_snapshot':checkpoint.parent.name,'checkpoint_front_sha256':checkpoint_front_sha256},'runs':{}}
 dump(out,result)
 for seed in map(int,a.seeds.split(',')):
  print('LOAD',a.dataset,a.alpha,seed,flush=True)
  concepts,test=F.load(a.dataset,a.alpha,seed,a.max_per_class)
  initial=concepts[0]['client_data'];site=max(range(len(initial)),key=lambda k:int((initial[k][1]==0).sum()))
  xb,yb=initial[site];ix=np.flatnonzero(yb==0);rng=np.random.default_rng([seed,9631]);ix=rng.choice(ix,min(a.context,len(ix)),replace=False)
  if len(ix)<2:raise RuntimeError('insufficient initial benign context')
  xc,yc=xb[ix],yb[ix];fe=FrontEnd(model,'cuda').fit(xc,yc)
  # All validation probes are initial training rows, not evaluation rows.
  q=xb[:min(12,len(xb))];e=fe.embed(q);e2=fe.embed(q)
  perm=np.arange(len(q))[::-1];ep=fe.embed(q[perm]);ec=np.concatenate([fe.embed(q[:len(q)//2]),fe.embed(q[len(q)//2:])])
  checks={'deterministic':bool(np.array_equal(e,e2)),'order_max_abs':float(np.max(np.abs(ep-e[perm]))),'batch_max_abs':float(np.max(np.abs(ec-e))),'finite':bool(np.isfinite(e).all())}
  if not checks['deterministic'] or not checks['finite'] or max(checks['order_max_abs'],checks['batch_max_abs'])>1e-4:raise RuntimeError(str(checks))
  d=xb.shape[1];proj=np.random.default_rng(20260928).normal(size=(e.shape[1],d)).astype('float32')/np.sqrt(d)
  scaler=StandardScaler().fit(fe.embed(xc)@proj)
  def transform(X):
   if not len(X):return np.empty((0,d),dtype=np.float32)
   z=np.concatenate([fe.embed(X[i:i+a.batch])@proj for i in range(0,len(X),a.batch)])
   z=scaler.transform(z).astype('float32')
   if not np.isfinite(z).all():raise RuntimeError('nonfinite features')
   return z
  tc=time.perf_counter();torch.cuda.reset_peak_memory_stats()
  transformed=[]
  for ci,c in enumerate(concepts):
   z={k:v for k,v in c.items() if k not in ['client_data','test']}
   z['client_data']=[(transform(x),y.copy()) for x,y in c['client_data']]
   z['test']=(transform(c['test'][0]),c['test'][1].copy());transformed.append(z)
   print('EMBED',seed,ci,c['name'],flush=True)
  et=(transform(test[0]),test[1].copy())
  features_s=time.perf_counter()-tc;peak=torch.cuda.max_memory_allocated()
  train_hash=digest(*[v for c in concepts for xy in c['client_data'] for v in xy]);test_hash=digest(*test)
  # No exact input row in held-out cohort may serve as context (duplicate features are rejected conservatively).
  eval_rows={hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest() for row in test[0]}
  overlap=sum(hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest() in eval_rows for row in xc)
  # Source identities are training-only by construction. Equal feature vectors need not be same records.
  rec={'context_site':site,'context_initial_indices':ix.tolist(),'context_hash':digest(xc,yc),'context_exact_feature_matches_in_test':overlap,'train_hash':train_hash,'test_hash':test_hash,'checks':checks,'feature_seconds':features_s,'feature_peak_cuda_bytes':peak,'context_payload_bytes':int(xc.nbytes+yc.nbytes),'context_broadcast_bytes':int((len(initial)-1)*(xc.nbytes+yc.nbytes)),'projection_hash':digest(proj),'feature_dim':d,'runs':{}}
  result['runs'][str(seed)]=rec;dump(out,result)
  for label,cc,tt in [('raw',concepts,test),('tabfm',transformed,et)]:
   assert np.array_equal(tt[1],test[1])
   for c,z in zip(concepts,cc):
    for (_,y),(_,yy) in zip(c['client_data'],z['client_data']):assert np.array_equal(y,yy)
   for arm in ['rehearsal','cda']:
    print('TRAIN',seed,label,arm,flush=True)
    rec['runs'][label+'_'+arm]=F.run_arm(a.dataset,arm,cc,tt,torch.device('cuda'),seed)
    dump(out,result)
    print('RESULT',seed,label,arm,rec['runs'][label+'_'+arm]['final']['macro_f1'],flush=True)
  print('DONE',a.dataset,seed,flush=True)
if __name__=='__main__':main()

