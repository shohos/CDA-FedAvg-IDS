#!/usr/bin/env bash
# Frozen protocol v2 (FINAL_EVAL_PROTOCOL_v2.md). Usage: bash run_final_v2.sh R1|R2
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
case "$1" in
  R1) SEEDS=47,48,49,50,51,52,53,54,55,56 ;;
  R2) SEEDS=600,601,602,603,604,605,606,607,608,609 ;;
  *) echo "usage: $0 R1|R2"; exit 1 ;;
esac
ARMS=iid_stream,nonstat,rehearsal,cda,cda_q1,fliids,fliids_norm,fliids_ce,fliids_ce_half,fliids_gb,fliids_ce_ls,fliids_ce_ls_norm
for DS in cicids unsw; do
  python final_eval.py --impl repaired --dataset $DS --alphas 0.1,0.5,1.0,100 --seeds $SEEDS \
    --arms $ARMS --fliids-lr 0.02 --out final_v2/$1_$DS.json
done
python final_eval.py --impl repaired --dataset ids2018 --alphas 0.5 --seeds $SEEDS \
  --arms $ARMS --fliids-lr 0.02 --out final_v2/$1_ids2018.json
echo "[$1 DONE]"
