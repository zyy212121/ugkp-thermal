"""Compare real resident execution with/without shared graph replay.
Uses identical table boundaries, variable transient intervals, three RK schemes,
and both frozen-mixture and single-gas paths. No benchmark timing is inferred.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, struct, subprocess
P=argparse.ArgumentParser();P.add_argument('--source',type=Path,required=True)
P.add_argument('--binaries',type=Path,required=True);P.add_argument('--output',type=Path,required=True)
P.add_argument('--solvers',nargs='+',default=['gasUGKP','gasUGKPSteady']);P.add_argument('--wall',action='store_true')
P.add_argument('--reference-binaries',type=Path,help='Compare the previous binaries against the candidate; both use graph execution')
P.add_argument('--csr-auto',action='store_true',help='Exercise dense two-phase automatic scheduling and post-transport moments')
a=P.parse_args()
if a.wall and a.csr_auto:P.error('--wall and --csr-auto exercise different supported gas capabilities')
a.output.mkdir(parents=True,exist_ok=False)
def entry(p,key,value):
 s,n=re.subn(r'\b'+re.escape(key)+r'\s+[^;]+;',key+' '+str(value)+';',p.read_text())
 if not n:s+='\n'+key+' '+str(value)+';\n'
 p.write_text(s)
def patch(c,field,name,body):
 p=c/'0'/field;s,n=re.subn(r'\b'+name+r'\s*\{[^}]*\}',name+' { '+body+' }',p.read_text());assert n==1;p.write_text(s)
def particles(c):
 entry(c/'constant/schedulingProperties','gpuResidentPureGasOnly','false')
 entry(c/'constant/schedulingProperties','gpuResidentParticleCapacity','50000')
 entry(c/'constant/schedulingProperties','gpuCsrLevel','auto' if a.csr_auto else 'L1')
 if a.csr_auto:entry(c/'constant/schedulingProperties','gpuCsrHeavyReductionAutoInterval',1)
 count=512 if a.csr_auto else 1
 for k,v in [('parcelMass','1e-10'),('particleTemperatureTransport','true'),('particleRho','2800'),('particleCp','800'),('particleTMin','1'),('particleTMax','10000')]:entry(c/'constant/particleProperties',k,v)
 p=c/'0/epsilonS';p.write_text(p.read_text().replace('internalField uniform 0;','internalField nonuniform List<scalar> 16 ('+' '.join([str(count*5.357142857142857e-10)]+['0']*15)+');'))
 (c/'0/Tp').write_text('FoamFile {version 2.0;format ascii;class volScalarField;object Tp;}\ndimensions [0 0 0 1 0 0 0]; internalField uniform 300; boundaryField { inlet {type fixedValue;value uniform 300;} outlet {type zeroGradient;} wall {type zeroGradient;} top {type symmetryPlane;} sides {type empty;} }\n')
 with (c/'0/gpuResidentStrictParticles.dat').open('wb') as f:
  f.write(f'UGKP_PARTICLES_SCHEMA1_BIN {count} {count}\n'.encode());f.write(struct.pack('<I',count))
  for v in [.25,.005,.005,70,0,0,500,0,2e-4,1e-10]:f.write(struct.pack('<'+'d'*count,*([v]*count)))
  for fmt,values in [('i',[0]*count),('i',[1]*count),('Q',[12345+i for i in range(count)]),('Q',list(range(1,count+1)))]:
   f.write(struct.pack('<'+fmt*count,*values))
def fields(c):
 times=sorted((p for p in c.iterdir() if p.is_dir() and re.fullmatch('[0-9.eE+-]+',p.name) and float(p.name)>0),key=lambda p:float(p.name))
 assert times,c
 output={}
 for t in times:
  for p in t.rglob('*'):
   if not p.is_file():continue
   try:s=p.read_text()
   except (UnicodeDecodeError,OSError):continue
   if p.name=='steadyWindowMean.csv':body=s
   elif 'internalField' in s and 'dimensions' in s:body=s[s.index('dimensions'):]
   else:continue
   nums=[float(v) for v in re.findall(r'(?<![A-Za-z_])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',body)]
   output[str(p.relative_to(c))]=nums
 # Sampling reports are part of the stopping decision and conservation audit.
 # Compare them as well as the persisted mean fields and physical state.
 selected=[]
 for line in (c/'run.log').read_text().splitlines():
  if any(mark in line for mark in [' meanChange=', '  windowStationary=',
      '  gasIntrinsicFvBoundaryExchange ', '  particleInjected ',
      '  particleTrackingBoundaryExchange ', '  particleInventoryMinusBoundaryExchange ']):
   selected.append(line)
 for index,line in enumerate(selected):
  output['sampling-report/'+str(index)]=[float(v) for v in re.findall(r'(?<![A-Za-z_])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',line)]
 return output
def particle_states(c):
 output={}
 for p in c.glob('*/gpuResidentStrictParticles.dat'):
  if float(p.parent.name)<=0:continue
  with p.open('rb') as f:
   schema,total,chunk=f.readline().decode().split();assert schema=='UGKP_PARTICLES_SCHEMA1_BIN'
   rows={}
   while len(rows)<int(total):
    count=struct.unpack('<I',f.read(4))[0];assert 0<count<=int(chunk)
    arrays=[struct.unpack('<'+fmt*count,f.read(struct.calcsize(fmt)*count)) for fmt in ['d']*10+['i','i','Q','Q']]
    for row in zip(*arrays):assert row[-1] not in rows;rows[row[-1]]=row
   assert not f.read(1)
  output[p.parent.name]=rows
 return output
results=[]
for solver in a.solvers:
 for mixture,parcels in ([(True,True)] if a.csr_auto else [(True,False)] if a.wall else [(False,False),(True,False),(True,True)]):
  for order,scheme in [(1,'Euler'),(2,'SSPRK2'),(3,'SSPRK3')]:
   pair=[];particle_pair=[];binary_hashes=[];label=f'{solver}-mix{int(mixture)}-particles{int(parcels)}-rk{order}'
   for graph in [0,1]:
    suffix=('reference' if graph==0 else 'candidate') if a.reference_binaries else f'graph{graph}'
    c=a.output/(label+'-'+suffix);c.mkdir()
    for part in ['0','constant','system']:shutil.copytree(a.source/'validation/shared_boundary/fixtures/gas-total'/part,c/part)
    if not mixture:
     for f in ['gasModelProperties','gasTransportProperties','gasWallProperties']:(c/'constant'/f).unlink(missing_ok=True)
    elif not a.wall:(c/'constant/gasWallProperties').unlink(missing_ok=True)
    p=c/'constant/fluidProperties'
    if a.wall:
     p.write_text(p.read_text().replace('model reactingSst; nodes 16;', 'model equilibriumOde; nodes 8;'))
     p=c/'constant/gasWallProperties';p.write_text(p.read_text().replace('0.01 0.02 0.1 0.4','0 0 0 0'))
     p=c/'constant/gasModelProperties';p.write_text(p.read_text().replace('coefficients (0 0);','coefficients (1e-5 1e-5);'))
    else:p.write_text(re.sub(r'turbulence\s*\{[\s\S]*','turbulence { simulationType laminar; }\n',p.read_text()))
    if parcels:particles(c)
    steady=solver=='gasUGKPSteady'
    for k,v in [('deltaT',1 if steady else '1e-8'),('endTime',6 if steady else '4e-7'),('adjustTimeStep','false' if steady else 'true'),('maxCo','.1'),('maxDeltaT','1e-7'),('writeControl','timeStep'),('writeInterval',1)]:entry(c/'system/controlDict',k,v)
    entry(c/'constant/schedulingProperties','gpuResidentMaxDeltaTGrowth','1.3')
    if steady:entry(c/'system/controlDict','boundaryTime','1.25e-7')
    p=c/'system/fvSchemes';p.write_text(re.sub(r'(ddtSchemes\s*\{\s*default\s+)\w+',r'\g<1>'+scheme,p.read_text()))
    patch(c,'T','inlet','type uniformFixedValue; uniformValue table ((0 300) (1e-6 400)); value uniform 300;')
    env=dict(os.environ,UGKP_GAS_GRAPH='1' if a.reference_binaries else str(graph),GAS_UGKP_CUDA_STACK_BYTES='65536',OMP_NUM_THREADS='2')
    binary=(a.reference_binaries if graph==0 and a.reference_binaries else a.binaries)/solver
    digest=hashlib.sha256()
    with binary.open('rb') as stream:
     for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    binary_hashes.append(digest.hexdigest())
    with (c/'run.log').open('w') as log:status=subprocess.run([str(binary),'-case',str(c)],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=120).returncode
    text=(c/'run.log').read_text();assert status==0,(label,graph,status,text[-4000:])
    pair.append(fields(c));particle_pair.append(particle_states(c) if parcels else {})
   assert pair[0].keys()==pair[1].keys(),(label,'field/time set differs')
   worst=0
   for key,left in pair[0].items():
    right=pair[1][key];assert len(left)==len(right),(label,key,'size')
    error=max((abs(x-y)/max(1,abs(x),abs(y)) for x,y in zip(left,right)),default=0)
    assert all(__import__('math').isfinite(v) for v in left+right),(label,key,'nonfinite')
    worst=max(worst,error)
    assert error<=(1e-10 if a.csr_auto else 1e-12),(label,key,error)
   assert particle_pair[0].keys()==particle_pair[1].keys(),(label,'particle time sets')
   particle_error=0
   for time,rows in particle_pair[0].items():
    other=particle_pair[1][time];assert rows.keys()==other.keys(),(label,time,'particle identities')
    for identity,left in rows.items():
     right=other[identity];assert left[10:]==right[10:],(label,time,identity,'particle cell/status/RNG')
     particle_error=max(particle_error,max(abs(x-y)/max(1,abs(x),abs(y)) for x,y in zip(left[:10],right[:10])))
   assert particle_error<=(1e-10 if a.csr_auto else 1e-12),(label,particle_error)
   results.append(dict(case=label,fields=len(pair[0]),max_scaled_difference=worst,max_particle_scaled_difference=particle_error,csr='auto' if a.csr_auto else 'L1',reference=str(a.reference_binaries) if a.reference_binaries else 'graph-disabled',binary_sha256=binary_hashes))
   (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
   print(label,'PASS',worst,flush=True)
