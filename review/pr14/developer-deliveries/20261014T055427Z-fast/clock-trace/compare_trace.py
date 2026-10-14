#!/usr/bin/env python3
"""Compare real CUDA solver's diagnostic F003 records bitwise, report by event."""
import argparse,collections,json,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);a=p.parse_args()
def read(path):
    occurrences=collections.Counter();rows={}
    for line in path.read_text().splitlines():
        if not line.startswith('F003,'):continue
        parts=line.split(',');key=tuple(parts[1:7]);n=occurrences[key];occurrences[key]+=1
        value=float.fromhex(parts[7]);rows[key+(n,)]=(parts[7],struct.pack('>d',value).hex())
    return rows
x,y=read(a.baseline),read(a.candidate)
summary={'baseline_records':len(x),'candidate_records':len(y),'missing_in_candidate':[], 'missing_in_baseline':[], 'first_difference_by_event':{},'difference_counts_by_event':{}}
counts=collections.Counter()
for key,v in x.items():
    if key not in y:summary['missing_in_candidate'].append(key);continue
    if v[1]==y[key][1]:continue
    event=key[3];counts[event]+=1
    if event not in summary['first_difference_by_event']:
        summary['first_difference_by_event'][event]={'window':key[0],'epoch':key[1],'microstep':key[2],'index':key[4],'field':key[5],'occurrence':key[6],'baseline_hex':v[0],'candidate_hex':y[key][0],'baseline_bits':v[1],'candidate_bits':y[key][1]}
summary['missing_in_baseline']=[k for k in y if k not in x]
summary['difference_counts_by_event']=dict(counts)
summary['all_records_bitwise_equal']=bool(x) and x==y
print(json.dumps(summary,indent=2))
