"""
verify_fingerprints.py -- compare the SHA-256 fingerprints recorded in result
files and launch records with the files in this directory.

Every final-evaluation result file stores the hash of each source file it ran
with (meta.config.code_sha256); the TabFM runs store meta.source_sha256; launch
records store the hash of the frozen protocol and of the code at launch.

    python verify_fingerprints.py                       # the main records
    python verify_fingerprints.py checkpoints/agg/fedavg_cicids.json ...
    python verify_fingerprints.py --code legacy/aggregation_study checkpoints/agg/fedavg_cicids.json

--code DIR is searched before the top level, for records written by code that has
since changed and is archived under legacy/.
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = ['checkpoints/final_eval_v2_launch.json', 'checkpoints/final_v2/R2_cicids.json',
           'checkpoints/agg_v3_launch.json', 'checkpoints/agg/fedavg_cicids.json',
           'checkpoints/final_v2/R2_ids2018_pooled.json',
           'checkpoints/tabfm_integration/tabfm_dev_cicids_42_46_a05.json']


CODE = []


def sha(name):
    for d in CODE + ['', 'docs/protocols']:
        p = os.path.join(HERE, d, name)
        if os.path.isfile(p):
            return hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return None


def recorded(rec):
    meta = rec.get('meta', {})
    hashes = dict(meta.get('config', {}).get('code_sha256') or meta.get('source_sha256')
                  or rec.get('code_sha256') or {})
    if rec.get('protocol'):
        hashes[rec['protocol']] = rec['protocol_sha256']
    return hashes


def main(paths):
    bad = 0
    for path in paths:
        print(path)
        for name, h in sorted(recorded(json.load(open(os.path.join(HERE, path)))).items()):
            now = sha(name)
            state = 'match' if now == h else ('missing' if now is None else 'CHANGED')
            bad += state != 'match'
            print(f'  {state:8} {name}')
    print(f'{bad} file(s) differ from the recorded fingerprints')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[:1] == ['--code']:
        CODE.append(args[1])
        args = args[2:]
    main(args or DEFAULT)
