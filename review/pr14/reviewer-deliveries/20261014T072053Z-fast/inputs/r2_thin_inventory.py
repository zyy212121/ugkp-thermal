from r2_runner import *
for app in ['CHT','FSH']:
 p=D/'inputs'/('actual-'+app+'-state-prefix.cuh');lines=p.read_text().splitlines();print('FILE',p)
 for i,x in enumerate(lines):
  if any(w in x for w in ['struct DeviceState','rhoE','gasStage','wallEnergy','WallHeat','Effusivity','solidProfile','particleEnthalpy','particleMass','particleTemperature','TotalEnergy','gasDt','TimeIntegrator','pT','pH','pM','Co','wallEnergy']):print(i+1,x)
 for name in ['CHT-coldWall2D-INPUT2','FSH-coldWall2D-INPUT2']:
  p=D/'artifacts/resident'/name
  if p.exists():print('fields',name,[str(q.relative_to(p)) for q in p.rglob('*') if q.is_file()][:100])
