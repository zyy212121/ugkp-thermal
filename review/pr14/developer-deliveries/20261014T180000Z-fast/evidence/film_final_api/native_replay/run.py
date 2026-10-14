from pathlib import Path
import subprocess,json,hashlib,struct,time
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014')
OLD=Path('/workspace/scratch/98b6339ec81b/pr14_fast4_evap_analysis')
checkpoint=ROOT/'review/pr14/reviewer-deliveries/20261014T072053Z-fast/artifacts/extra/wet-evaporation-S209-D204/checkpoint-1'
cmd=['g++','-std=c++17','-O0','-g0','-ffp-contract=off','-DUGKWP_GAS_SPECIES=2','-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),str(HERE/'checkpoint_film_boundary_probe.cpp'),str(ROOT/'applications/CHMT/film/CpuFilmDriver.C'),'-o',str(HERE/'probe')]
receipt=[]
for label,args in [('build',cmd),('run',[str(HERE/'probe'),str(checkpoint),str(HERE/'current_state.json')])]:
 start=time.monotonic()
 with (HERE/(label+'.stdout')).open('w') as out,(HERE/(label+'.stderr')).open('w') as err:r=subprocess.run(args,stdout=out,stderr=err)
 receipt.append(dict(label=label,argv=args,exit=r.returncode,seconds=time.monotonic()-start))
 (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2))
 if r.returncode:raise SystemExit(r.returncode)
def load(p):return {v['label']:v for v in map(json.loads,p.read_text().splitlines())}
def exact(v):
 if isinstance(v,float):return ('f64',struct.pack('>d',v).hex())
 if isinstance(v,dict):return tuple((k,exact(x)) for k,x in sorted(v.items()))
 if isinstance(v,list):return tuple(map(exact,v))
 return (type(v).__name__,v)
def physical(v):return exact({k:v[k] for k in ('film','filmAux')})
a=load(OLD/'boundary_original_state.json.snapshots.jsonl');b=load(HERE/'current_state.json.snapshots.jsonl');oldCandidate=load(OLD/'boundary_signed_open_edge_state.json.snapshots.jsonl')
shared=[];promoted=[]
for label,value in a.items():assert physical(value)==physical(b[label]),label;shared.append(label)
for mode in ('checkpoint_aux','checkpoint_bulk'):
 for half in (0,1,10,30):
  old=f'{mode}-complete-half{half}';new=f'{mode}-advance-half{half}'
  assert physical(a[old])==physical(b[new]),(old,new)
  promoted.append([old,new])
for label,value in oldCandidate.items():assert physical(value)==physical(b[label]),('previous signed candidate',label)
paths=[Path(__file__).resolve(),HERE/'checkpoint_film_boundary_probe.cpp',HERE/'current_state.json.snapshots.jsonl',OLD/'boundary_original_state.json.snapshots.jsonl',OLD/'boundary_signed_open_edge_state.json.snapshots.jsonl',ROOT/'applications/CHMT/film/CpuFilmDriver.C',ROOT/'applications/CHMT/film/CpuFilmDriver.H']
result={'scope':'Current native CPU checkpoint/operator replay, frozen geometry/recorded forcing; no full GPU owner replay. Only actual film and filmAux fields compared; deleted report/checksum excluded.','shared_success_count':len(shared),'promoted_count':len(promoted),'previous_signed_success_count':len(oldCandidate),'shared':shared,'promoted':promoted,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(HERE/'physical_comparison.json').write_text(json.dumps(result,indent=2))
print(f'PASS: {len(shared)} original successes; {len(promoted)} promoted original-terminal/current-ordinary; {len(oldCandidate)} previous signed successes; all film/filmAux fields binary64 exact')
