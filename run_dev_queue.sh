#!/usr/bin/env bash
# Audit-repair development runs, one after another on the GPU (consumed seeds only).
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
D=audit_repair_dev
# 1. legacy must reproduce the saved FL-IIDS numbers (they used lr 0.02)
python final_eval.py --impl legacy --dataset cicids --alphas 0.5 --seeds 47 --arms fliids,fliids_ce --fliids-lr 0.02 --out $D/legacy_repro_cicids_fliids.json
# 2. the repaired implementation, all arms and loss variants
ARMS=iid_stream,nonstat,rehearsal,cda,cda_q1,fliids,fliids_ce,fliids_ce_half,fliids_gb,fliids_ce_ls
python final_eval.py --impl repaired --dataset cicids --alphas 0.1,0.5 --seeds 42,43,44 --arms $ARMS --fliids-lr 0.02 --out $D/repaired_cicids.json
python final_eval.py --impl repaired --dataset unsw --alphas 0.1,0.5 --seeds 42,43,44 --arms $ARMS --fliids-lr 0.02 --out $D/repaired_unsw.json
python final_eval.py --impl repaired --dataset ids2018 --alphas 0.5 --seeds 42,43,44 --arms $ARMS --fliids-lr 0.02 --out $D/repaired_ids2018.json
# 3. which repair moved which number
python dev_repairs.py --dataset cicids --seeds 42,43,44 --out $D/decomp_cicids.json
python dev_repairs.py --dataset unsw --seeds 42,43,44 --out $D/decomp_unsw.json
python dev_repairs.py --dataset ids2018 --seeds 42,43 --out $D/decomp_ids2018.json
echo "[QUEUE DONE]"
