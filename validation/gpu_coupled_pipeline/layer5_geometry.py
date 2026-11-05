#!/usr/bin/env python3
from pathlib import Path
import json,math,hashlib,argparse
parser=argparse.ArgumentParser(description='Compare frozen and GPU native geometry logs and executable provenance.')
parser.add_argument('output',type=Path)
E=parser.parse_args().output.resolve()
a=(E/'baseline.log').read_text().splitlines();b=(E/'candidate.log').read_text().splitlines();assert len(a)==len(b)
errors={};cases=[]
for left,right in zip(a,b):
 x,y=left.split(),right.split();assert x and y and x[0]==y[0]
 if x[0] in ['STATUS','IDENTITY','CASES']:
  assert left==right,(left,right)
  if x[0]=='STATUS':cases.append(dict(id=int(x[1]),accepted=bool(int(x[2])),numerical_failure=bool(int(x[3])),message=' '.join(x[4:])))
 else:
  assert len(x)==len(y),(x[0],len(x),len(y));old=list(map(float,x[1:]));new=list(map(float,y[1:]));assert all(math.isfinite(v) for v in old+new)
  error=max((abs(p-q)/max(1.,abs(p),abs(q)) for p,q in zip(old,new)),default=0.);errors[x[0]]=max(errors.get(x[0],0),error)
  assert error<(2e-13 if x[0]=='wallDistance' else 1e-15),(x[0],error)
report=dict(status='PASS',scope='Actual frozen88bc versus native CUDA geometry, default round-to-nearest. Host floating-point exception flags are not emulated by device arithmetic; unchanged CPU frontend tests retain their host fenv contract.',cases=cases,maximum_scaled_field_errors=errors,binaries={n:hashlib.sha256((E/n).read_bytes()).hexdigest() for n in ['baseline','candidate']})
(E/'results.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(cases),'native geometry cases',errors)