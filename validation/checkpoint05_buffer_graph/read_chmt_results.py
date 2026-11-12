#!/usr/bin/env python3
"""Read completed batch only; never launches solver/profiler or declares exit0 convergence."""
import argparse,csv,json,math,re
from pathlib import Path
def number(s):
    try:return float(s)
    except (ValueError,TypeError):return None
def csv_rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def summaries(rows):
    result={}
    for key in (rows[0] if rows else []):
        v=[number(row.get(key)) for row in rows];v=[x for x in v if x is not None]
        if v:result[key]={'count':len(v),'finite':all(math.isfinite(x) for x in v),'min':min(v),'max':max(v),'sum':sum(v),'mean':sum(v)/len(v)}
    return result
def api_stats(prefix):
    found=[]
    for p in prefix.parent.glob(prefix.name+'-api*.csv'):
        # nsys may put informational text before the CSV header.
        lines=p.read_text().splitlines();start=next((i for i,l in enumerate(lines) if 'Name' in l and ('Num Calls' in l or 'Instances' in l)),None)
        if start is None:continue
        for row in csv.DictReader(lines[start:]):
            name=row.get('Name','')
            if any(x in name for x in ['cudaMalloc','cudaFree','cudaGraphInstantiate','cudaStreamBeginCapture','cudaGraphExecKernelNodeSetParams','cudaGraphLaunch']):found.append(row)
    return found

def main():
    a=argparse.ArgumentParser();a.add_argument('manifest',type=Path);a.add_argument('--output',type=Path,required=True);args=a.parse_args();manifest=json.loads(args.manifest.read_text());out={'nodes':{},'comparisons':{},'limitations':['Counts aggregate graph types; use trace event attribution before claiming gas-only capture counts.','Single profiled run timing is indicative, not a repeated throughput estimate.','Finite fields/exit0 alone do not prove nonlinear or energy convergence.','State binary is not decoded by this reader; temperature/energy/geometry observations use the actual published CSV.']};payload={}
    for n in manifest['nodes']:
        case=Path(n['case']);folder=case/'chmtOutput';checkpoints=sorted(int(p.name.split('-')[-1]) for p in folder.glob('checkpoint-*') if p.is_dir())
        record={'expectedWindows':n['expectedWindows'],'checkpoints':checkpoints,'terminalSequence':max(checkpoints,default=None),'matchesExpectedWindowCount':max(checkpoints,default=None)==n['expectedWindows'],'csv':{},'cudaApi':api_stats(Path(n['profilePrefix']))};payload[n['id']]={}
        for p in sorted(folder.glob('*.csv')):
            rows=csv_rows(p);record['csv'][p.name]=summaries(rows);payload[n['id']][p.name]=rows
        log=Path(n['log']);text=log.read_text(errors='replace') if log.exists() else ''
        record['diagnosticLines']=[l for l in text.splitlines() if any(k in l.lower() for k in ['warning','fatal','converged=','residual','accepted window'])]
        record['observedLogExists']=log.exists();out['nodes'][n['id']]=record
    for kind in ['static','moving']:
      for bits in [32,64]:
        b=payload[kind+'-baseline'+str(bits)];c=payload[kind+'-candidate'+str(bits)];comparison={}
        for name,rows in b.items():
            other=c.get(name)
            if other is None or len(rows)!=len(other):comparison[name]={'shapeMatch':False};continue
            diff={}
            for key in (rows[0] if rows else []):
                pairs=[(number(x.get(key)),number(y.get(key))) for x,y in zip(rows,other)];pairs=[(x,y) for x,y in pairs if x is not None and y is not None]
                if pairs:diff[key]={'absoluteMax':max(abs(y-x) for x,y in pairs),'scaledMax':max(abs(y-x)/max(1,abs(x),abs(y)) for x,y in pairs),'sumDifference':sum(y-x for x,y in pairs)}
            comparison[name]={'shapeMatch':True,'fields':diff}
        out['comparisons'][kind+'-'+str(bits)]=comparison
    def safe(value):
        if isinstance(value,float) and not math.isfinite(value):return str(value)
        if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
        if isinstance(value,list):return [safe(v) for v in value]
        return value
    args.output.write_text(json.dumps(safe(out),indent=2,allow_nan=False))
if __name__=='__main__':main()
