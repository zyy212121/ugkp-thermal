from r2_runner import *
import sys
sys.path.insert(0,str(ROOT/'common'));from thermal_particle_restart_current import iter_chunks
for name in ['CHT-cold2D-fullgroups-S211','CHT-cold2D-explicit16-S213','FSH-cold2D-explicit16-S213']:
 case=ROOT/'work/cases/resident'/name;print('CASE',name)
 print('points paths',*[str(p.relative_to(case)) for p in case.rglob('points')],sep='\n')
 for q in [case/'system/blockMeshDict',case/'constant/polyMesh/boundary',case/'1e-06/rho',case/'0/rho']:
  print(str(q.relative_to(case)),q.read_text()[:1800] if q.exists() else 'missing')
 p=case/'0/gpuResidentStrictParticles.dat'
 if 'S213' in name:
  marker,data=next(iter_chunks(p));print('INPUT',marker,{k:data[k].tolist() for k in data if k in ['px','py','pz'] or 'cell' in k.lower() or 'id' in k.lower() or k=='pStuck'})
 print('records particle0 rawfiles',list((case/'0').iterdir()))
