#!/usr/bin/env python3
"""Acceptance of actual native solver output, never generates CFD data."""
import json,pathlib,sys,csv,math
root=pathlib.Path(sys.argv[1]); results=[]
eps=sys.float_info.epsilon
negative=json.loads(next(root.glob('incompatible_*/negative.json')).read_text()); assert negative['passed'] and negative['statePreserved']
for p in sorted(root.glob('*/summary.json')):
 r=json.loads(p.read_text()); r['case']=p.parent.name; results.append(r)
 r['fields']=[(z['component'],float(z['value']),float(z['scale'])) for z in csv.DictReader((p.parent/'endpoint.csv').open())]
 r['trajectory']=[list(map(float,z)) for z in csv.reader((p.parent/'state_trajectory.csv').open())]
 assert r['trajectory'] and all(len(row)==len(r['fields'])+1 and all(math.isfinite(x) for x in row) for row in r['trajectory']), p
 assert all(b[0]>a[0] for a,b in zip(r['trajectory'],r['trajectory'][1:])), p
 assert r['completed'] and r['gasSteps']>20, p
 assert r['maxRelativeGcl']<512*eps and r['maxNativeVolumeRelative']<512*eps and r['maxNativeSweepRelative']<512*eps, p
 assert r['maxCfl']<.45 and r['minGasTemperature']>100 and r['minDensity']>0, p
 assert r['maxRelativeSpeciesSumError']<512*eps and r['maxThermalDiffusionNumber']<.45, p
 assert r['gasCells']>=48 and r['solidCells']>=16, p
 if r['mode']=='gcl': assert r['maxFreestreamError']<1e-10, p
 else:
  assert r['maxRelativeMassResidual']<512*eps and r['maxRelativeEnergyResidual']<512*eps, p
  assert r['cumulativeRelativeMassResidual']<512*eps and r['cumulativeRelativeEnergyResidual']<512*eps, p
  assert r['maxSweepRemainderRelative']<512*eps and r['maxPackingVolumeRelative']<512*eps, p
  assert r['restartError']==0 and r['rollbackChecked'] and r['acceptedWindows']>=20, p
 if r['mode']=='moving':
  assert r['recession']>1e-9 and r['massTransferred']>0 and abs(r['pressureWork'])>1e-12, p
  assert r['maxOuterNormalSpeed']>0 and abs(r['externalPressureWork'])>0, 'moving outer-wall work branch must be exercised'
  fronts=[v for c,v,scale in r['fields'] if c=='front']; assert len(fronts)>=4 and max(fronts)-min(fronts)>1e-8, 'actual nonuniform 3-D recession required'
  assert r['maxMomentumResidual']<1e-10 and r['executedGasSteps']>=r['gasSteps'] and r['executedMaterialSolves']>=r['acceptedWindows'], p
assert len(results)==7, 'all seven prescribed cases must run'
def select(dt,win): return next(r for r in results if r['mode']=='moving' and abs(r['microDt']-dt)<1e-15 and abs(r['windowDt']-win)<1e-15)
def norm(a,b):
 assert len(a['fields'])==len(b['fields'])
 for (name,x,scale),(name2,y,scale2) in zip(a['fields'],b['fields']):
  assert name==name2 and math.isfinite(x) and math.isfinite(y) and math.isfinite(scale) and scale>0 and scale==scale2
 return math.sqrt(sum(((x-y)/scale)**2 for (name,x,scale),(name2,y,scale2) in zip(a['fields'],b['fields']))/len(a['fields']))
def trajectory_norm(a,b):
 pairs=[]
 for first in a['trajectory']:
  second=min(b['trajectory'],key=lambda v:abs(v[0]-first[0]))
  if abs(first[0]-second[0])>1e-12:continue
  assert len(first)==len(second)
  pairs.append(math.sqrt(sum((x-y)**2 for x,y in zip(first[1:],second[1:]))/(len(first)-1)))
 assert pairs
 return max(pairs)
def ratio(a,b,c,fn):
 coarse=fn(a,b);fine=fn(b,c);assert coarse>1e-12, ('refinement signal below roundoff',coarse)
 return fine/coarse
coarse,mid,fine=[select(dt,4e-6) for dt in [4e-7,2e-7,1e-7]]
gasRatio=ratio(coarse,mid,fine,norm)
gasTrajectoryRatio=ratio(coarse,mid,fine,trajectory_norm)
coupling=[select(1e-7,w) for w in [4e-6,2e-6,1e-6]]
windowRatio=ratio(*coupling,norm)
windowTrajectoryRatio=ratio(*coupling,trajectory_norm)
assert gasRatio<.85, ('gas timestep refinement did not decrease',gasRatio)
assert windowRatio<.85, ('coupling refinement did not decrease',windowRatio)
assert gasTrajectoryRatio<.85 and windowTrajectoryRatio<.85, ('trajectory refinement',gasTrajectoryRatio,windowTrajectoryRatio)
for r in results:r.pop('fields');r.pop('trajectory')
report={'incompatibleQuadRejection':negative,'gasTrajectoryRatio':gasTrajectoryRatio,'couplingTrajectoryRatio':windowTrajectoryRatio,'passed':True,'gasRefinementRatio':gasRatio,'couplingRefinementRatio':windowRatio,'results':results}
(root/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'gasRefinementRatio':gasRatio,'couplingRefinementRatio':windowRatio}))
