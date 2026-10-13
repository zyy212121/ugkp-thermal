import pathlib,shutil,re,json
from run_fast import ROOT,D,run,cmd,amend,sha
B=ROOT/'work/baseline-native';binary=ROOT/'work/install/base/bin/CHT'
t=run('BUILD-BASE-CHT',cmd('env',f'FOAM_USER_APPBIN={binary.parent}','bash',B/'applications/CHT/tools/build_cuda_solver.sh'),[12,24,33],900,'Baseline d611 original native CHT with matched sm89/O3/fmadfalse, isolated binary.',inputs=[B/'applications/CHT/tools/build_cuda_solver.sh'],kind='BUILD',variant='baseline')
if t['exit_code']==0:
 src=D/'inputs/native/resident/CHT-coldWall2D-INPUT2';case=ROOT/'work/baseline-cold2D';shutil.copytree(src,case)
 t=run('MESH-BASE-COLD2D',cmd('blockMesh','-case',case),[33],300,'Identical real case for historical producer/rejection qualification.',inputs=list(src.rglob('*')),kind='SETUP',variant='baseline')
 if t['exit_code']==0:
  t=run('BASE-COLD2D',cmd(binary,'-case',case),[12,26,33],300,'Same disabled-solid candidate wall; check native original failure vs current numerical publication, not assume old-success regression.',inputs=list(src.rglob('*')),binary=binary,variant='baseline')
  amend(t['id'],verdict='FAIL' if t['exit_code'] else 'INCONCLUSIVE',observations=(D/t['log']).read_text()[-6000:])
# Full native restart: time5e-6 actual particle/field/residual/epsilon-history publication.
names=['gasUGKP-periodic-PHYS1','FSH-periodic-INPUT1-PHYS1','CHT-periodic-INPUT2-PHYS1','CHT-coldWall2D-P007']
for name in names:
 app=name.split('-')[0];original=ROOT/'work/cases/resident'/name
 inputname='CHT-coldWall2D-INPUT2' if 'P007' in name else name
 src=D/'inputs/native/resident'/inputname;case=ROOT/'work/cases/resident'/(name+'-RESUME')
 shutil.copytree(src,case);shutil.copytree(original/'5e-06',case/'5e-06');shutil.copytree(original/'constant/polyMesh',case/'constant/polyMesh')
 p=case/'system/controlDict';s=p.read_text();s=re.sub(r'startTime\s+0;',r'startTime 5e-6;',s);p.write_text(s)
 snapshot=D/'inputs/native/resident'/case.name;shutil.copytree(case,snapshot)
 binary=ROOT/('work/install/P007/bin/CHT' if 'P007' in name else 'work/install/ns2/bin/'+app)
 variant='diagnostic-P007' if 'P007' in name else 'original'
 t=run('RESTART-NATIVE-'+name,cmd(binary,'-case',case),[5,12,18,19,26,29]+([33] if 'coldWall' in name else []),300,'Consume actual saved native codec fields, float profile plus double carry/history, real restart owner; compare all final fields and every decoded particle member.',inputs=list(snapshot.rglob('*')),binary=binary,variant=variant)
 out=D/'artifacts/resident'/case.name;out.mkdir(parents=True,exist_ok=True)
 for q in case.iterdir():
  try:v=float(q.name)
  except ValueError:continue
  if v>5e-6:shutil.copytree(q,out/q.name)
 amend(t['id'],artifacts=[str(out.relative_to(D))])
