from pathlib import Path
import re,json,csv,hashlib,sys
import numpy as np
r=Path(sys.argv[1]).resolve(); requested_sequence=int(sys.argv[2])
original=r/'case-pr41-fp64'
labels=['pr41-fp64','candidate-fp64','candidate-fp32']
sys.path.insert(0,str(r.parent/'source/candidate/tool/restart'))
import thermal_particle_restart_current as particle_wire
def particle_stats(path):
 chunks=[d for marker,d in particle_wire.iter_chunks(path)]
 if not chunks:return {'count':0,'massKg':0}
 arrays={k:np.concatenate([d[k] for d in chunks]) for k in chunks[0]}
 for key,value in arrays.items():
  if value.dtype.kind=='f' and not np.isfinite(value).all():raise ValueError('nonfinite particle '+key)
 m=arrays['pm'];T=arrays['pT'];v=np.column_stack([arrays[k] for k in ('pux','puy','puz')]);mass=float(m.sum())
 return {'count':len(m),'massKg':mass,'massWeightedTemperatureK':float(np.dot(m,T)/mass) if mass else None,
  'temperatureQuantilesK':np.quantile(T,[0,.1,.5,.9,1]).tolist(),'kineticJ':float(np.dot(m,(v*v).sum(axis=1))*.5),
  'momentumKgMS':(m[:,None]*v).sum(axis=0).tolist(),'stuckCount':int(np.count_nonzero(arrays['pStuck']))}
def clean(s):
 return re.sub(r'//[^\n]*','',re.sub(r'/\*.*?\*/','',s,flags=re.S))
def rawlist(p):
 t=clean(p.read_text());m=re.search(r'\n\s*(\d+)\s*\n\s*\(\s*\n(.*)\n\s*\)\s*$',t,re.S)
 if not m:raise ValueError(str(p))
 return int(m.group(1)),m.group(2)
def mesh(region):
 d=original/'constant'/region/'polyMesh'
 n,p=rawlist(d/'points');pts=np.fromstring(p.replace('(',' ').replace(')',' '),sep=' ').reshape(n,3)
 _,f=rawlist(d/'faces');faces=[np.fromstring(s,sep=' ',dtype=int) for s in re.findall(r'\d+\s*\(([^)]*)\)',f)]
 _,o=rawlist(d/'owner');owner=np.fromstring(o,sep=' ',dtype=int)
 _,n=rawlist(d/'neighbour');neigh=np.fromstring(n,sep=' ',dtype=int)
 nc=max(owner.max(),neigh.max())+1;vol=np.zeros(nc)
 area=[];fc=[]
 for i,indices in enumerate(faces):
  vertices=pts[indices];av=np.cross(vertices,np.roll(vertices,-1,axis=0)).sum(axis=0)/2
  area.append(np.linalg.norm(av));fc.append(vertices.mean(axis=0))
  signed=sum(np.dot(vertices[0],np.cross(vertices[k],vertices[k+1]))/6 for k in range(1,len(vertices)-1))
  vol[owner[i]]+=signed
  if i<len(neigh):vol[neigh[i]]-=signed
 bd=clean((d/'boundary').read_text());patches={}
 for name,block in re.findall(r'(\w+)\s*\{(.*?)\}',bd,re.S):
  nf=re.search(r'nFaces\s+(\d+)',block);st=re.search(r'startFace\s+(\d+)',block)
  if nf and st:patches[name]=(int(st.group(1)),int(nf.group(1)))
 return {'volume':vol,'area':np.array(area),'centers':np.array(fc),'patches':patches,'nCells':int(nc),'minVolume':float(vol.min()),'totalVolume':float(vol.sum())}
meshes={reg:mesh(reg) for reg in ['fluid','graphite']}
def field(path,region,patch=None):
 t=clean(path.read_text());vector='volVectorField' in t
 if patch:
  m=re.search(r'\b'+re.escape(patch)+r'\s*\{(.*?)\}',t,re.S)
  if not m:raise ValueError('missing patch '+patch)
  t=m.group(1);count=meshes[region]['patches'][patch][1];prefix=r'\bvalue'
 else:count=meshes[region]['nCells'];prefix=r'\binternalField'
 m=re.search(prefix+r'\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;',t,re.S)
 if m:
  a=np.fromstring(m.group(2).replace('(',' ').replace(')',' '),sep=' ')
  return a.reshape(int(m.group(1)),3) if vector else a
 m=re.search(prefix+r'\s+uniform\s+(\([^)]*\)|[^;]+);',t,re.S)
 if not m:raise ValueError('no field data '+str(path))
 a=np.fromstring(m.group(1).replace('(',' ').replace(')',' '),sep=' ')
 return np.tile(a,(count,1)) if vector else np.full(count,a[0])
def state(path):
 t=path.read_text();d={}
 for k,v in re.findall(r'(?m)^\s*(\w+)\s+([^;]+);',t):
  v=v.strip('" \n')
  try:v=float(v)
  except ValueError:pass
  d[k]=v
 return d
def stats(a,w):
 s=np.linalg.norm(a,axis=1) if a.ndim==2 else a
 return {'min':float(s.min()),'max':float(s.max()),'weightedMean':float(np.dot(s,w)/w.sum()),'weightedRms':float(np.sqrt(np.dot(s*s,w)/w.sum())),'finite':bool(np.isfinite(a).all())}
def errors(a,b,w):
 diff=a-b;d=np.linalg.norm(diff,axis=1) if diff.ndim==2 else np.abs(diff)
 ref=np.linalg.norm(b,axis=1) if b.ndim==2 else np.abs(b)
 den=np.dot(ref*ref,w)
 return {'absoluteMax':float(d.max()),'weightedRmse':float(np.sqrt(np.dot(d*d,w)/w.sum())),'relativeL2':float(np.sqrt(np.dot(d*d,w)/den)) if den else None,'relativeL1':float(np.dot(d,w)/np.dot(ref,w)) if np.dot(ref,w) else None,'weightedMeanDifference':float(np.dot(a-b,w)/w.sum()) if a.ndim==1 else None}
wallStart,wallCount=meshes['fluid']['patches']['fluid_to_graphite'];wallW=meshes['fluid']['area'][wallStart:wallStart+wallCount]
wc=meshes['fluid']['centers'][wallStart:wallStart+wallCount];wallS=np.r_[0,np.cumsum(np.linalg.norm(np.diff(wc,axis=0),axis=1))]
field_specs=[('fluid/'+n,'fluid',None) for n in ['rho','p','T','U','epsilonS','Tp','theta','rhoE','rhoEs','rhoHp','k','omega','particleRadiationAbsorptionCoefficient','particleRadiationScatteringCoefficient','particleRadiationExtinctionCoefficient','particleRadiationAsymmetryFactor']]+[('graphite/T','graphite',None)]+[('fluid/'+n,'fluid','fluid_to_graphite') for n in ['T','gasConvectiveWallHeatFlux','particleRadiationWallHeatFlux','particleReflectedWallHeatFlux','particleStuckWallHeatFlux']]
summary={'mesh':{reg:{k:v for k,v in m.items() if k in ['nCells','minVolume','totalVolume','patches']} for reg,m in meshes.items()},'runs':{},'comparisons':{},'limitations':['One serial run per variant: timings are indicative, not a repeated-run confidence interval.','PR41 FP64 is a version-comparison reference, not an exact solution or independent accuracy truth.','Same particle random seed, but FP32 changes the floating-point trajectory and may alter particle sampling/locations.','Same coupling cadence, initial conditions and tolerances. Adaptive timestep schedules may differ; compare matching exchange sequence and report actual times.','Restart from0.29500248649130967, sequence59; endTime0.30500348649130967. Compare common outputs within the requested interval; sequence61 may not be written. Radiation occurs at60 and includes the carried radiation interval beginning0.29000244439008815.']}
allfields={};profile_rows=[]
for label in labels:
 c=r/('case-'+label);paths=sorted([p for p in c.iterdir() if p.is_dir() and re.fullmatch(r'\d+\.\d+',p.name)],key=lambda x:float(x.name))
 rec={'exchanges':[]};allfields[label]={}
 log=r/(label+'.log')
 logged_times=re.findall(r'(?m)^(?:Time = |UGKP_REPRO step=\d+ time=)([0-9.eE+-]+)',log.read_text(errors='replace')) if log.exists() else []
 rec['terminalLoggedTime']=float(logged_times[-1]) if logged_times else None
 rec['terminalLoggedTimeOffsetFromNominalEnd']=rec['terminalLoggedTime']-0.305 if rec['terminalLoggedTime'] is not None else None
 rec['terminalLoggedTimeScope']='Last available progress time, not guaranteed terminal step; logs are sampled and final fields may be unwritten.'
 for out in paths:
  sp=out/'thermalExchangeState'
  if not sp.exists():continue
  st=state(sp);seq=int(st['exchangeSequence'])
  if seq<=59 or float(out.name)>0.30500348649130967:continue
  event={'path':str(out),'time':float(out.name),'state':st,'fields':{}};allfields[label][seq]={}
  part=out/'gpuResidentStrictParticles.dat'
  if part.exists():
   with part.open('rb') as fh:event['particleHeader']=fh.readline().decode(errors='replace').strip()
   event['particleStatistics']=particle_stats(part)
  for name,region,patch in field_specs:
   key=name+('@'+patch if patch else '')
   fp=out/name
   if not fp.exists():continue
   try:a=field(fp,region,patch)
   except Exception as e:event['fields'][key]={'parseError':str(e)};continue
   w=wallW if patch else meshes[region]['volume'];allfields[label][seq][key]=(a,w)
   s=stats(a,w)
   if patch and 'HeatFlux' in name:s['integralW']=float(np.dot(a,w))
   event['fields'][key]=s
  if 'fluid/T@fluid_to_graphite' in allfields[label][seq]:
   for i in range(wallCount):
    row={'label':label,'sequence':seq,'time':float(out.name),'wall_s_m':float(wallS[i]),'area_m2':float(wallW[i])}
    for key,(a,w) in allfields[label][seq].items():
     if '@' in key:row[key]=float(a[i])
    profile_rows.append(row)
  if all(key in allfields[label][seq] for key in ('fluid/rho','fluid/rhoE','fluid/epsilonS')):
   get=lambda name:allfields[label][seq][name][0]
   epsg=1-get('fluid/epsilonS');volume=meshes['fluid']['volume']
   event['inventoryObservations']={'gasMassKg':float(np.dot(epsg*get('fluid/rho'),volume)),
    'gasEnergyJ':float(np.dot(epsg*get('fluid/rhoE'),volume)),
    'particleMechanicalJ':float(np.dot(get('fluid/rhoEs'),volume)),
    'particleThermalJ':float(np.dot(get('fluid/rhoHp'),volume)),
    'scope':'Published epsilonS/rhoE/rhoEs/rhoHp volume integral; external flow and solid fixed-value boundary time-integrals not saved. This is not full-domain energy closure.'}
  rec['exchanges'].append(event)
 lp=r/(label+'.log')
 if lp.exists():
  text=lp.read_text(errors='replace');rec['progress']=[dict(zip(['step','time','dt','elapsed'],map(float,v))) for v in re.findall(r'UGKP_REPRO step=(\d+) time=([^ ]+) dt=([^ ]+) elapsed=([^\n]+)',text)]
  rec['warningsCount']=text.count('FOAM Warning');rec['fatalCount']=text.count('FATAL');rec['warningLines']=[ln for ln in text.splitlines() if any(x in ln.lower() for x in ['warning','budget exhausted','error','failed'])][:40]
  rec['precisionLines']=[ln for ln in text.splitlines() if 'precision' in ln.lower()][:8]
 gp=r/(label+'-gpu.csv')
 if gp.exists():
  lines=gp.read_text().splitlines()[1:];rows=[]
  for line in lines:
   cols=line.split(',')
   if len(cols)<8:continue
   try:rows.append([float(re.search(r'[\d.]+',cols[i]).group()) for i in [2,3,4,5,6]])
   except (ValueError,AttributeError):continue
  if rows:
   x=np.array(rows);rec['gpu']={k:{'mean':float(x[:,j].mean()),'max':float(x[:,j].max()),'min':float(x[:,j].min())} for j,k in enumerate(['utilPercent','memoryMiB','tempC','powerW','smMHz'])}
 tp=r/(label+'.time')
 if tp.exists():
  timetext=tp.read_text();mm=re.search(r'Elapsed \(wall clock\) time .*?: ([0-9:.]+)',timetext)
  if mm:
   parts=[float(n) for n in mm.group(1).split(':')];rec['gnuTimeWallSeconds']=sum(v*60**i for i,v in enumerate(parts[::-1]))
  mm=re.search(r'Maximum resident set size \(kbytes\): (\d+)',timetext)
  if mm:rec['maxHostRssKiB']=int(mm.group(1))
 cycle={};prevRad=0.29000244439008815
 for event in rec['exchanges']:
  dt=event['state']['completedSimulationTimeS']-event['state']['previousExchangeSimulationTimeS'];rt=event['state']['completedRadiationSimulationTimeS']
  for name in ['gasConvectiveWallHeatFlux','particleReflectedWallHeatFlux','particleStuckWallHeatFlux','particleRadiationWallHeatFlux']:
   key='fluid/'+name+'@fluid_to_graphite';fields=event['fields']
   if key not in fields or 'integralW' not in fields[key]:continue
   window=(rt-prevRad) if name=='particleRadiationWallHeatFlux' else dt
   cycle[name]=cycle.get(name,0.)+fields[key]['integralW']*window
  prevRad=rt
 rec['cumulativeWallEnergyJ']=cycle
 rec['actualStopping']={'status':'Manual review of all warnings required',
  'budgetExhausted':'budget exhausted' in lp.read_text(errors='replace').lower() if lp.exists() else None,
  'warningFile':str(r/(label+'-execution.json')),
  'strictConvergenceClaim':False}
 summary['runs'][label]=rec
for test,ref in [('candidate-fp64','pr41-fp64'),('candidate-fp32','pr41-fp64'),('candidate-fp32','candidate-fp64')]:
 pair={}
 for seq,values in allfields[test].items():
  if seq not in allfields[ref]:continue
  pair[seq]={k:errors(a,allfields[ref][seq][k][0],w) for k,(a,w) in values.items() if k in allfields[ref][seq]}
  for key,(a,w) in values.items():
   if key in pair[seq] and '@' in key and 'HeatFlux' in key:
    b=allfields[ref][seq][key][0];aInt=float(np.dot(a,w));bInt=float(np.dot(b,w));pair[seq][key].update({'testIntegralW':aInt,'referenceIntegralW':bInt,'relativeIntegralDifference':(aInt-bInt)/bInt if bInt else None})
 summary['comparisons'][test+'_vs_'+ref]=pair
summary['requestedSequence']=requested_sequence
summary['commonSequences']=sorted(set.intersection(*(set(allfields[label]) for label in labels)))
summary['endTime']=0.30500348649130967
summary['nominalRequestedEndTime']=0.305
summary['endTimePolicy']='Common stopping upper bound allows sequence61 near 0.305002; actual output times and deviations from nominal 0.305 must be reported.'
summary['missingSequencePolicy']='Unwritten sequence61 is unavailable evidence, not a failed calculation. Compare common saved exchanges and report final log times separately.'
summary['manualPhysicalReviewRequired']=True
summary['budgetScope']={'closedTests':'Independent closed gas/coupling/sparse/CHMT fixtures provide dimensional closure.',
 'openCase':'Full gas mass/energy and fixed-value solid boundary integrated fluxes are unavailable with original probes=0; do not equate inventory change with conservation error.',
 'species':'Original case has a single gas; no multi-species density fields are present.'}
summary['reviewHints']=[]
for pair,events in summary['comparisons'].items():
 for seq,observations in events.items():
  for name,v in observations.items():
   if v.get('relativeL2') is not None and v['relativeL2']>.05:
    summary['reviewHints'].append({'pair':pair,'sequence':seq,'field':name,'relativeL2':v['relativeL2'],
     'meaning':'5% review hint only, never an acceptance or rejection threshold.'})
for label,rec in summary['runs'].items():
 seqs=[int(e['state']['exchangeSequence']) for e in rec['exchanges']]
 rec['requestedSequenceAvailable']=requested_sequence in seqs
 rec['latestSavedSequence']=max(seqs) if seqs else None
 rec['latestSavedTime']=max((e['time'] for e in rec['exchanges']),default=None)
 rec['terminalStateLimitation']='Latest saved exchange is not necessarily the solver terminal state; no forced extra write or extra continuation.'
 for event in rec['exchanges']:
  for name,values in event['fields'].items():
   if values.get('finite') is False:raise RuntimeError(label+' nonfinite '+name)
if (r/'status.json').exists():summary['runStatus']=json.loads((r/'status.json').read_text())
(r/('metrics-seq'+str(requested_sequence)+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
if profile_rows:
 with (r/('wall-profiles-seq'+str(requested_sequence)+'.csv')).open('w') as f:
  cols=sorted(set().union(*(row.keys() for row in profile_rows)));w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(profile_rows)
print(json.dumps({'runStatus':summary.get('runStatus'),'exchanges':{k:[{'time':e['time'],'sequence':e['state']['exchangeSequence'],'lastRadiation':e['state']['completedRadiationSimulationTimeS'],'particleHeader':e.get('particleHeader')} for e in v['exchanges']] for k,v in summary['runs'].items()}},indent=2))
