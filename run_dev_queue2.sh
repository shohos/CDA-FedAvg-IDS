#!/usr/bin/env bash
# Second development queue: the normalised-target FL-IIDS readings (consumed seeds only).
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
D=audit_repair_dev
A=fliids_norm,fliids_ce_ls_norm
python final_eval.py --impl repaired --dataset cicids --alphas 0.1,0.5 --seeds 42,43,44 --arms $A --fliids-lr 0.02 --out $D/repaired_cicids_norm.json
python final_eval.py --impl repaired --dataset unsw --alphas 0.1,0.5 --seeds 42,43,44 --arms $A --fliids-lr 0.02 --out $D/repaired_unsw_norm.json
python final_eval.py --impl repaired --dataset ids2018 --alphas 0.5 --seeds 42,43,44 --arms $A --fliids-lr 0.02 --out $D/repaired_ids2018_norm.json
echo "[QUEUE2 DONE]"
