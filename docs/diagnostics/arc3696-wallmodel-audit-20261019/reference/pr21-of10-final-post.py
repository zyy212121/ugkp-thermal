from pathlib import Path
import subprocess,json,time
B=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr21-audit-execution-20261019');O=Path('/mnt/d/tempt/pr21-wallmodel-audit-20261019/of10-reference');rows=[]
for ny in [80,160]:
 for work in [False,True]:
  case=B/('of10-reference-'+('work-'if work else'')+str(ny));tool=str(B/'isolated-appbin/rhoSimpleWallAuditFoam')if work else'rhoSimpleFoam'
  cmd=['bash','-c',f'source /opt/openfoam10/etc/bashrc >/dev/null 2>&1; export FOAM_USER_APPBIN="{B}/isolated-appbin"; export FOAM_USER_LIBBIN="{B}/isolated-libbin"; exec {tool} -case "{case}" -postProcess -latestTime -funcs "(wallShearStress yPlus)"']
  t=time.time()
  with(O/(case.name+'-final-post.log')).open('w')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=60)
  rows.append({'argv':cmd,'exit':r.returncode,'elapsed_seconds':time.time()-t});assert r.returncode==0,rows[-1]
(O/'final-post-commands.json').write_text(json.dumps(rows,indent=2));print('final-post',len(rows),'passed')
