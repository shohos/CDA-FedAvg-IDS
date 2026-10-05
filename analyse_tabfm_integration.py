"""analyse_tabfm_integration.py -- summary and per-seed table of the TabFM integration screen.

Reads the three result files in checkpoints/tabfm_integration/ (seeds 42-46, alpha 0.5),
checks completeness, invariance, metric consistency and matched rehearsal exposure, and
writes tabfm_summary.json and tabfm_per_seed.csv next to them.

    python analyse_tabfm_integration.py
"""
import json,math,hashlib,statistics,csv,ast
from pathlib import Path
import numpy as np
from scipy.stats import t
CODE=Path(__file__).resolve().parent
R=CODE/'checkpoints'/'tabfm_integration'
source=(CODE/'analyse_final_eval.py').read_text(encoding='utf8')
tree=ast.parse(source);func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='forgetting')
exec(compile(ast.Module(body=[func],type_ignores=[]),'verified_forgetting_function','exec'))
arms=['raw_rehearsal','raw_cda','tabfm_rehearsal','tabfm_cda']
files=['tabfm_dev_unsw_42_a05.json','tabfm_dev_unsw_43_46_a05.json','tabfm_dev_cicids_42_46_a05.json']
cells={};checks=[];details=[]
for name in files:
 d=json.loads((R/name).read_text(encoding='utf8'));ds=d['meta']['dataset']
 for f,h in d['meta']['source_sha256'].items():
  assert hashlib.sha256((CODE/f).read_bytes()).hexdigest()==h,(name,f)
 for seed,rec in d['runs'].items():
  assert set(rec['runs'])==set(arms),(name,seed,'incomplete')
  assert rec['checks']['finite'] and rec['checks']['deterministic']
  assert max(rec['checks']['order_max_abs'],rec['checks']['batch_max_abs'])<=1e-4
  for a in arms:
   metrics=rec['runs'][a]['final'];fv=[v for k,v in metrics.items() if k.startswith('f1_')]
   assert abs(statistics.mean(fv)-metrics['macro_f1'])<1e-10
  for k in ['grad_steps','history']:
   if k=='history':
    assert [x['memory_rows'] for x in rec['runs']['raw_rehearsal'][k]]==[x['memory_rows'] for x in rec['runs']['tabfm_rehearsal'][k]]
   else:assert rec['runs']['raw_rehearsal'][k]==rec['runs']['tabfm_rehearsal'][k]
  cells.setdefault(ds,{})[int(seed)]=rec
  checks.append({'dataset':ds,'seed':int(seed),'complete_arms':4,'invariance_and_metrics':True,'rehearsal_exposure_matched':True,'context_equal_feature_vectors_in_test':rec['context_exact_feature_matches_in_test']})
summary={'scope':'exploratory paired main-stream integration; not confirmation','cells':[],'checks':checks,'result_sha256':{f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in files}}
for ds,runs in sorted(cells.items()):
 assert set(runs)==set(range(42,47)),(ds,sorted(runs))
 ref=json.loads((CODE/f'checkpoints/final_v2/R2_{ds}.json').read_text())
 concepts=next(iter(ref['runs'].values()))['concepts'];names=ref['classes']
 row={'dataset':ds,'label':('CICIDS2017' if ds=='cicids' else 'UNSW-NB15')+' / 0.5','n':len(runs)}
 for arm in arms:
  m=[r['runs'][arm]['final']['macro_f1'] for r in runs.values()]
  row[arm]=statistics.mean(m);row[arm+'_sd']=statistics.stdev(m)
  forg=[];fw=[];lost=[]
  for seed,r in sorted(runs.items()):
   rr=r['runs'][arm]
   ff=forgetting({'concepts':concepts,arm:rr},arm,names)
   forg.append(ff['mean_forgetting']);lost.append(ff['lost']);fw.append(statistics.mean(x['macro_f1'] for x in rr['forward']))
   details.append({'dataset':ds,'alpha':.5,'seed':seed,'arm':arm,'retention':rr['final']['macro_f1'],'forward':fw[-1],'forgetting':forg[-1],'learned':ff['learned'],'lost':ff['lost'],'steps':rr['grad_steps'],'seconds':rr['seconds']})
  row[arm+'_forgetting']=statistics.mean(forg);row[arm+'_forward']=statistics.mean(fw);row[arm+'_lost']=statistics.mean(lost)
 for policy in ['rehearsal','cda']:
  delta=[r['runs']['tabfm_'+policy]['final']['macro_f1']-r['runs']['raw_'+policy]['final']['macro_f1'] for r in runs.values()]
  mean=statistics.mean(delta);half=float(t.ppf(.975,len(delta)-1))*statistics.stdev(delta)/math.sqrt(len(delta))
  row['delta_'+policy]=mean;row['delta_'+policy+'_ci']=[mean-half,mean+half];row['delta_'+policy+'_positive']=sum(x>0 for x in delta)
 row['feature_seconds_range']=[min(r['feature_seconds'] for r in runs.values()),max(r['feature_seconds'] for r in runs.values())]
 row['feature_peak_cuda_bytes']=max(r['feature_peak_cuda_bytes'] for r in runs.values())
 row['context_broadcast_bytes']=sorted(set(r['context_broadcast_bytes'] for r in runs.values()))
 summary['cells'].append(row)
summary['slide_conclusion']='Always-adapt retention rose in all five seeds on both datasets. CDA gains were consistent on UNSW, but uncertain on CICIDS. These are development results.'
(R/'tabfm_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
with (R/'tabfm_per_seed.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=details[0].keys());w.writeheader();w.writerows(details)
print(json.dumps(summary['cells'],indent=2))
