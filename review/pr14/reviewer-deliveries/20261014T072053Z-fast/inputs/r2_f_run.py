import json,shutil
from r2_runner import *
for name in ['wet-film','wet-evaporation','dry-ablation']:
 case=ROOT/'work/cases/extra'/name;b=ROOT/'work/install/ns2/bin/CHMT';inp=D/'inputs/native/extra'/name
 t=run('EXTRA-'+name,cmd(b,'-case',case),[3,6,8,9,10,13,14,20,21,22,26,29],expected='Actualnative shortGPUgas/originalCPUmaterial film evaporation ablation; rawinventory/stage/budget retained, independentclosure follows.',inputs=[p for p in inp.rglob('*') if p.is_file()],binary=b)
 if (case/'chmtOutput').exists():shutil.copytree(case/'chmtOutput',D/'artifacts/extra'/name)
for name in ['CHT-actual-coupler','CHT-actual-radiation']:
 case=ROOT/'work/cases/resident'/name;inp=D/'inputs/native/resident'/name
 for region in [None,'solid']:
  args=['blockMesh','-case',case]+(['-region',region] if region else [])
  t=run('MESH-'+name+('-'+region if region else ''),cmd(*args),[18,23,26,30,33],expected='Real blockmesh from recordednewvalid nativeinput, coupledpatchmatchedgeometry.',inputs=[case/'system'/('solid/blockMeshDict' if region else 'blockMeshDict')],kind='SETUP');assertion(t,t['exit_code']==0,'RealOpenFOAMmesh setup '+str(t['exit_code']))
 b=ROOT/'work/install/ns2/bin/CHT';t=run('NATIVE-'+name,cmd(b,'-case',case),[18,19,20,23,26,30,33],expected='ActualCHTnativeGPU owner +real solidcoupler, selectedsinglecoupledface/mixeddefaults; activeDO radiationtrue inone case andfalsecontrol. ScopefiniteIO/resource/inventory auditseparate.',inputs=[p for p in inp.rglob('*') if p.is_file()],binary=b)
 save_case(case,name)
