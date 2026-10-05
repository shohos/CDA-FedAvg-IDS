"""
merge_v2_parts.py -- merges a run set's CSE-CIC-IDS2018 results that were computed
in two processes (seed halves) into one file. Refuses unless both halves were
produced with identical settings and code (meta.config) and share no seed.
The second half is moved to final_v2/parts/ as a record, not deleted.

    python merge_v2_parts.py R1 R2
"""
import json, os, shutil, sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'final_v2')

for R in sys.argv[1:]:
    main_p, part_p = os.path.join(D, f'{R}_ids2018.json'), os.path.join(D, f'{R}_ids2018_b.json')
    a, b = json.load(open(main_p)), json.load(open(part_p))
    assert a['meta']['config'] == b['meta']['config'], f'{R}: settings or code differ between halves'
    overlap = set(a['runs']) & set(b['runs'])
    assert not overlap, f'{R}: both halves contain {sorted(overlap)}'
    a['runs'].update(b['runs'])
    a['meta']['seeds'] = sorted(set(a['meta']['seeds']) | set(b['meta']['seeds']))
    a['meta']['merged_from'] = [os.path.basename(main_p), os.path.basename(part_p)]
    a['meta']['command_part_b'] = b['meta']['command']
    json.dump(a, open(main_p, 'w'), indent=1)
    os.makedirs(os.path.join(D, 'parts'), exist_ok=True)
    shutil.move(part_p, os.path.join(D, 'parts', os.path.basename(part_p)))
    arms = {k for c in a['runs'].values() for k in c if k != 'concepts'}
    print(f'{R}: merged, {len(a["runs"])} cells, seeds {a["meta"]["seeds"]}, {len(arms)} arms')
