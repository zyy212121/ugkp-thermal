from pathlib import Path
import shutil,subprocess,json,time,sys,hashlib
B=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr21-audit-execution-20261019')
O=Path('/mnt/d/tempt/pr21-wallmodel-audit-20261019/of10-reference')
src=B/'rhoSimpleWallAuditFoam'
if not src.exists():
 shutil.copytree('/opt/openfoam10/applications/solvers/compressible/rhoSimpleFoam',src)
 p=src/'Make/files';p.write_text('rhoSimpleFoam.C\n\nEXE = $(FOAM_USER_APPBIN)/rhoSimpleWallAuditFoam\n')
 p=src/'EEqn.H';s=p.read_text();assert s.count('+ thermophysicalTransport->divq(he)')==1
 s=s.replace('+ thermophysicalTransport->divq(he)','+ thermophysicalTransport->divq(he)\n      + fvc::div(turbulence->devTau() & U)')
 p.write_text(s)
 q=subprocess.run(['bash','-c',f'source /opt/openfoam10/etc/bashrc >/dev/null 2>&1; export FOAM_USER_APPBIN="{B}/isolated-appbin"; export FOAM_USER_LIBBIN="{B}/isolated-libbin"; mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN"; wmake "{src}"'],capture_output=True,text=True)
 (O/'viscous-work-build.log').write_text(q.stdout+q.stderr)
 (O/'viscous-work-source.diff').write_text(subprocess.run(['diff','-u','/opt/openfoam10/applications/solvers/compressible/rhoSimpleFoam/EEqn.H',str(src/'EEqn.H')],capture_output=True,text=True).stdout)
 assert q.returncode==0,q.stderr[-2000:]
ny=int(sys.argv[1]);old=B/f'of10-reference-{ny}';case=B/f'of10-reference-work-{ny}'
if not case.exists():
 for sub in ['0','constant','system']:shutil.copytree(old/sub,case/sub)
else:
 assert not [p for p in case.iterdir()if p.is_dir()and p.name.isdigit()and p.name!='0'], 'Do not overwrite computed reference'
 log=O/f'{ny}-viscous-work.log'
 if log.exists():shutil.copy2(log,O/f'{ny}-viscous-work-initial-missing-scheme.log')
p=case/'system/controlDict';p.write_text(p.read_text().replace('application rhoSimpleFoam','application rhoSimpleWallAuditFoam'))
p=case/'system/fvSchemes';p.write_text(p.read_text().replace(' default none;',' default none;\n div((devTau&U)) Gauss linear;'))
cmd=['bash','-c',f'source /opt/openfoam10/etc/bashrc >/dev/null 2>&1; export FOAM_USER_APPBIN="{B}/isolated-appbin"; export FOAM_USER_LIBBIN="{B}/isolated-libbin"; exec "{B}/isolated-appbin/rhoSimpleWallAuditFoam" -case "{case}"']
t=time.time()
with(O/f'{ny}-viscous-work.log').open('w')as f:q=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1500)
binary=B/'isolated-appbin/rhoSimpleWallAuditFoam'
(O/f'{ny}-viscous-work-command.json').write_text(json.dumps({'command':cmd,'exit':q.returncode,'elapsed_seconds':time.time()-t,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()},indent=2))
print('exit',q.returncode,'seconds',time.time()-t)
sys.exit(q.returncode)
