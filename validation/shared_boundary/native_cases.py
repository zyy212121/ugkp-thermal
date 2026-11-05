import os,shutil,re,subprocess,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
mode=sys.argv[2] if len(sys.argv)>2 else 'gas'
def patch(c,field,name,body):
 p=c/'0'/field;s,n=re.subn(r'\b'+name+r'\s*\{[^}]*\}',name+' { '+body+' }',p.read_text());assert n==1,(p,name);p.write_text(s)
def entry(c,key,value):
 p=c/'system/controlDict';s,n=re.subn(r'\b'+key+r'\s+[^;]+;',key+' '+str(value)+';',p.read_text());assert n==1,(key,n);p.write_text(s)
def clone(src,name):
 c=out/name;c.mkdir()
 for part in ['0','constant','system']:shutil.copytree(src/part,c/part)
 return c
def run(c,binary):
 env=dict(os.environ,GAS_UGKP_CUDA_STACK_BYTES='65536',CHT_CUDA_STACK_BYTES='65536',CHMT_CUDA_STACK_BYTES='65536',OMP_NUM_THREADS='4')
 with (c/'run.log').open('w') as log:status=subprocess.run([str(R/'bin'/binary),'-case',str(c)],stdout=log,stderr=subprocess.STDOUT,env=env).returncode
 print(c.name,status,flush=True)
 if status:print((c/'run.log').read_text()[-4500:]);raise RuntimeError(c)
 return status
results={}
if mode=='gas':
 src=Path(__file__).resolve().parent/'fixtures/gas-total'
 for solver in ['gasUGKP','gasUGKPSteady']:
  for variant in ['constant','table','polynomial']:
   c=clone(src,solver+'-'+variant)
   if solver=='gasUGKPSteady':entry(c,'endTime',2);entry(c,'deltaT',1)
   if variant=='table':
    patch(c,'T','inlet','type uniformFixedValue; uniformValue table ((0 300) (1e-7 310) (2e-7 320)); value uniform 300;')
    patch(c,'U','inlet','type uniformFixedValue; uniformValue table ((0 (70 0 0)) (2e-7 (80 0 0))); value uniform (70 0 0);')
   if variant=='polynomial':
    patch(c,'T','inlet','type uniformFixedValue; uniformValue polynomial ((300 0) (1e8 1)); value uniform 300;')
   results[c.name]=run(c,solver)
else:
 for solver,name in [('CHT','cht-mixture'),('CHMT','chmt-mixture')]:
  if mode not in ['coupled',solver]:continue
  src=Path(__file__).resolve().parent/'fixtures'/('cht' if solver=='CHT' else 'chmt')
  for variant in ['legacy','constant','table']:
   c=clone(src,solver+'-'+variant)
   if variant!='legacy':
    p=c/'constant/polyMesh/boundary';s=p.read_text();s=re.sub(r'(right\s*\{[^}]*?type\s+)wall',r'\1patch',s);p.write_text(s)
    patch(c,'U','right','type pressureInletOutletVelocity; value uniform (-1 0 0);')
    p=c/'0/U';s=re.sub(r'internalField\s+uniform\s+\([^)]*\);','internalField uniform (-1 0 0);',p.read_text());p.write_text(s)
    T=600 if solver=='CHMT' else 300
    # Model-specific reference pressure only affects the verification input.
    pressure=170000 if solver=='CHMT' else 101325
    if not (c/'0/p').exists():
     (c/'0/p').write_text('FoamFile {version 2.0;format ascii;class volScalarField;object p;}\ndimensions [1 -1 -2 0 0 0 0];\ninternalField uniform '+str(pressure)+';\nboundaryField {left {type zeroGradient;} right {type zeroGradient;} sides {type zeroGradient;}}\n')
    patch(c,'p','right',f'type totalPressure; p0 uniform {pressure}; psi thermo:psi; gamma 1.4; value uniform {pressure};')
    patch(c,'T','right',f'type totalTemperature; T0 uniform {T}; gamma 1.4; value uniform {T};')
    if variant=='table':patch(c,'T','right',f'type uniformFixedValue; uniformValue table ((0 {T}) (0.001 {T+5})); value uniform {T};')
   results[c.name]=run(c,solver)
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
