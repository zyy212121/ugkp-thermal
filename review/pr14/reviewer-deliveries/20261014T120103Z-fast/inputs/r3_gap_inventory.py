from r3_runner import *
for folder,words in [('applications/CHT',['mapper','selectedPatches','coupledPatches','empty()','malloc']),('applications/CHMT/tests',['Js','prepareWall','MassPublished']),('common',['mapping','class SharedGasStorageOwner'])]:
 print('SEARCH',folder)
 count=0
 for p in (ROOT/folder).rglob('*'):
  if not p.is_file() or p.suffix not in ['.C','.H','.py','.cuh']:continue
  if folder=='applications/CHT' and not any(w in p.name for w in ['Coupler','Thermal','Wire','Mapping']):continue
  if folder=='common' and not any(w in p.name for w in ['SharedGasStorage','Coupler','Wire']):continue
  lines=p.read_text(errors='replace').splitlines()
  for i,l in enumerate(lines):
   if any(w in l for w in words):
    print(str(p.relative_to(ROOT)),i+1,'\n'+'\n'.join(f'{j+1}: {lines[j]}' for j in range(max(0,i-1),min(len(lines),i+3))));count+=1
    if count>70:break
  if count>70:break
for rel in ['work/cases/resident/CHT-active16-ASCII-S306/constant/thermalExchangeProperties','tests/test_chmt_zero_phase_material.py']:
 p=ROOT/rel
 if p.exists():print('FILE',rel,p.read_text()[:6600])
print('HOSTTESTS',[str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('*') if any(w in p.name for w in ['wire','mapping','coupler','resource','wall'])])
