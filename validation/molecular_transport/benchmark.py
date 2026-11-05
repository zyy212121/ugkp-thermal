#!/usr/bin/env python3
"""Alternating end-to-end timings; mesh generation is excluded, startup/I/O included."""
import argparse
import json
from pathlib import Path
import statistics
import native_matrix as matrix

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir',type=Path,required=True)
    parser.add_argument('--baseline-bin-dir',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    assert args.repeats>=3
    work=args.work.resolve();work.mkdir(parents=True,exist_ok=False)
    models=json.loads((matrix.HERE/'fixtures.json').read_text())
    results={}
    for name,end,dt in [('gas-wave',.001,1e-5),('gas-wall',1e-5,2.5e-7),('cht-contact',.0004,1e-5),('chmt-mixture',.005,1e-5)]:
        meta=models[name];solver=meta['solver'];samples={key:[] for key in ('baseline','const','sutherland')}
        terminals={}
        for repeat in range(args.repeats+1):
            order=list(samples)
            if repeat%2:order.reverse()
            for mode in order:
                case=work/f'{name}-{repeat}-{mode}'
                matrix.prepare(matrix.HERE/'fixtures'/name,case,meta,mode!='sutherland')
                control=case/'system/controlDict'
                matrix.entry(control,'endTime',end);matrix.entry(control,'deltaT',dt)
                if solver!='CHMT':
                    matrix.entry(control,'maxDeltaT',dt);matrix.entry(control,'writeInterval',end)
                binary=(args.baseline_bin_dir if mode=='baseline' else args.bin_dir).resolve()/solver
                elapsed=matrix.run([binary,'-case',case],case/'run.log')
                if repeat:samples[mode].append(elapsed)
                terminals[mode]=case
        # Check that the timing extension preserves const parity too.
        parity=matrix.compare_chmt(terminals['baseline'],terminals['const']) if solver=='CHMT' else matrix.compare_fields(matrix.times(terminals['baseline'])[-1],matrix.times(terminals['const'])[-1])
        results[name]={'end_time':end,'max_dt':dt,'seconds':samples,'median_seconds':{key:statistics.median(value) for key,value in samples.items()},'const_parity':parity}
        (work/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,results[name]['median_seconds'],flush=True)
if __name__=='__main__':main()
