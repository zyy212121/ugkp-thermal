import pathlib,shutil,json,re
from run_fast import ROOT,D,run,cmd,amend
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n最后追加sanitizer与profile继续：original CHMT Ns2 thermal 8+8两窗口mem/init/sync；P007+P010 cold2D mem/init/sync/race、真实5e-6重启至1e-5全fields比较。P010 cold1D endTime1e-4共100微步让ring0进入wet导热，物性/dt不改，writeInterval1e-5仅输出节奏，300s/例，不算原版通过。\n')
for name,src,binary,variant,tools in [
 ('CHMT-thermal',D/'inputs/native/chmt/thermal-full',ROOT/'work/install/ns2/bin/CHMT','original',['memcheck','initcheck','synccheck']),
 ('CHT-cold2D-P010',D/'inputs/native/resident/CHT-coldWall2D-INPUT2',ROOT/'work/install/P010/bin/CHT','diagnostic-P007+P010',['memcheck','initcheck','synccheck','racecheck'])]:
 for tool in tools:
  case=ROOT/'work/final-sanitizer'/(name+'-'+tool);shutil.copytree(src,case)
  if name.startswith('CHMT'):
   mesh=ROOT/'work/cases/chmt/thermal-full'
   for relative in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(mesh/relative,case/relative)
  else:shutil.copytree(ROOT/'work/cases/resident/CHT-coldWall2D-P010/constant/polyMesh',case/'constant/polyMesh')
  t=run('SAN-FINAL-'+name+'-'+tool,cmd('compute-sanitizer','--tool',tool,'--target-processes','all','--error-exitcode','86',binary,'-case',case),[12,26,28,29]+([33] if name.startswith('CHT-') else [3,5,8,9,14]),300,'Actual native fresh-case state owners, tool-specific summary0, numerical validity checked independently.',inputs=list(src.rglob('*')),binary=binary,variant=variant,kind='SANITIZER')
  raw=(D/t['log']).read_text();zero='ERROR SUMMARY: 0 errors' in raw or tool=='racecheck' and 'RACECHECK SUMMARY: 0 hazards' in raw
  amend(t['id'],verdict='PASS' if t['exit_code']==0 and zero else 'FAIL',observations=raw[-4000:])
name='CHT-coldWall2D-P010';src=D/'inputs/native/resident/CHT-coldWall2D-INPUT2';full=ROOT/'work/cases/resident'/name;case=ROOT/'work/cases/resident'/(name+'-RESUME')
shutil.copytree(src,case);shutil.copytree(full/'5e-06',case/'5e-06');shutil.copytree(full/'constant/polyMesh',case/'constant/polyMesh');p=case/'system/controlDict';p.write_text(re.sub(r'startTime\s+0;',r'startTime 5e-6;',p.read_text()))
snapshot=D/'inputs/native/resident'/case.name;shutil.copytree(case,snapshot);binary=ROOT/'work/install/P010/bin/CHT'
t=run('RESTART-NATIVE-'+name,cmd(binary,'-case',case),[5,18,19,26,29,33],300,'Patched latest profile/carry consume actual5e-6 restart, all final fields compared.',inputs=list(snapshot.rglob('*')),binary=binary,variant='diagnostic-P007+P010')
out=D/'artifacts/resident'/case.name;out.mkdir(parents=True,exist_ok=True);shutil.copytree(case/'1e-05',out/'1e-05')
checker=D/'inputs/resident_check_fp64_mean.py';check=run('COMPARE-RESTART-'+name,cmd('python3',checker,'restart',name),[5,18,19,26,29,33],300,'Full particle members bitwise plus final fields byte equality.',inputs=[checker],kind='ANALYSIS',variant='diagnostic-P007+P010')
amend(check['id'],verdict='PASS' if check['exit_code']==0 else 'FAIL',observations=(D/check['log']).read_text());amend(t['id'],verdict='PASS' if check['exit_code']==0 else 'FAIL',checker_result_id=check['id'])
src=D/'inputs/native/resident/CHT-coldWall1D-INPUT2';case=ROOT/'work/cases/resident/CHT-coldWall1D-P010-WET100';shutil.copytree(src,case);shutil.copytree(ROOT/'work/cases/resident/CHT-coldWall1D-P010/constant/polyMesh',case/'constant/polyMesh');p=case/'system/controlDict';s=p.read_text().replace('endTime 1e-5;','endTime 1e-4;').replace('writeInterval 1e-6;','writeInterval 1e-5;');p.write_text(s)
snapshot=D/'inputs/native/resident'/case.name;shutil.copytree(case,snapshot)
t=run('CHT-COLD1D-P010-WET100',cmd(binary,'-case',case),[12,23,26,33],300,'100 microsteps through ring0 first wet contact, unchanged physical parameters/dt, full written profile/carry/caloric consistency.',inputs=list(snapshot.rglob('*')),binary=binary,variant='diagnostic-P007+P010')
out=D/'artifacts/resident'/case.name;out.mkdir(parents=True,exist_ok=True)
for q in case.iterdir():
 try:v=float(q.name)
 except ValueError:continue
 if v>0:shutil.copytree(q,out/q.name)
check=run('FIELDS3-'+case.name,cmd('python3',checker,'finite',case.name),[23,26,33],300,'Same mean double/inversion abs1e-9, all actual physical particles and timestamps finite.',inputs=[checker],kind='ANALYSIS',variant='diagnostic-P007+P010')
amend(check['id'],verdict='PASS' if check['exit_code']==0 else 'FAIL',observations=(D/check['log']).read_text());amend(t['id'],verdict='PASS' if check['exit_code']==0 else 'FAIL',checker_result_id=check['id'])
