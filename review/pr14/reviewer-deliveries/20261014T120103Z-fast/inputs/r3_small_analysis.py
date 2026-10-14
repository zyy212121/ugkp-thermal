from r3_runner import *
for p in (ROOT/'common').glob('*Cold*'):
 print('COLDFILE',p.name)
 if p.is_file():
  ls=p.read_text().splitlines()
  for i,l in enumerate(ls):
   if any(k in l for k in ['__global__','groupSize','groupId','threadIdx.x','8;','8)','8,','blockDim']):print(i+1,l)
for name in ['onecell-melting-S308','onecell-solidification-S308']:
 raw=(D/'logs'/('PHASE-D302-'+name+'.log')).read_text();print('PHASE',name,'\n'.join(l for l in raw.splitlines() if l.startswith(('R3PHASE','R3CONFIG','CHMT accepted'))))
for app in ['CHT','FSH']:
 report=json.loads((D/'artifacts/resident'/(app+'-active16-ASCII-S306-active-profile-audit.json')).read_text());print('COLDERROR',app,report['maximum_active_error_K'])
for pos in ['pre','post']:
 a=[l for l in (D/'logs/THERMAL-SEED-thermal-full.log').read_text().splitlines() if l.startswith('R3SEED_CELL,'+pos+',2,1,')];b=[l for l in (D/'logs/THERMAL-SEED-thermal-restart.log').read_text().splitlines() if l.startswith('R3SEED_CELL,'+pos+',2,1,')];print('THERMALSEED',pos,[(x,y) for x,y in zip(a,b) if x!=y])
print('BYTE_PAIRED',[(q['id'],q['verdict']) for q in json.loads((D/'results.json').read_text())['tests'] if 'S308' in q['id']])
