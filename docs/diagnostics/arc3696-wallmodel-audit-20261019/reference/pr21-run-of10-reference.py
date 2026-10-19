from pathlib import Path
import subprocess,time,json,sys,os,hashlib
base=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr21-audit-execution-20261019')
out=Path('/mnt/d/tempt/pr21-wallmodel-audit-20261019/of10-reference')
ny=int(sys.argv[1]);case=base/f'of10-reference-{ny}'
env=os.environ.copy();env['OMP_NUM_THREADS']='1'
commands=[]
for tool,args in [('blockMesh',[]),('checkMesh',[]),('rhoSimpleFoam',[])]:
 log=out/f'{ny}-{tool}.log'
 cmd=['bash','-c','source /opt/openfoam10/etc/bashrc >/dev/null 2>&1; export FOAM_USER_APPBIN="'+str(base/'isolated-appbin')+'"; export FOAM_USER_LIBBIN="'+str(base/'isolated-libbin')+'"; exec '+tool+' -case "'+str(case)+'"']
 start=time.time()
 with log.open('w') as f:
  try: result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=1500);code=result.returncode
  except subprocess.TimeoutExpired:code=124
 commands.append({'argv':cmd,'exit':code,'elapsed_seconds':time.time()-start,'log':str(log)})
 (out/f'{ny}-commands.json').write_text(json.dumps(commands,indent=2))
 print(json.dumps(commands[-1]),flush=True)
 if code:sys.exit(code)
print('COMPLETE',flush=True)
