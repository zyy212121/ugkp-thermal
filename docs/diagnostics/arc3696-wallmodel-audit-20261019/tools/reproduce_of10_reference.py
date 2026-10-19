#!/usr/bin/env python3
"""Independent steady OF10 references; not identical gasUGKP/RK replay.
Uses pressure-based OF10 SST with native Spalding/omega/Jayatilleke wall laws.
"""
from pathlib import Path
import argparse,subprocess,os,json,hashlib,shutil,time
from of10_reference_case import writecase

def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshots',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--openfoam',type=Path,default=Path('/opt/openfoam10'));p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);source=a.snapshots.resolve()/'wallFunction-f9';assert(source/'system/blockMeshDict').is_file();of=a.openfoam.resolve();logs=out/'logs';logs.mkdir();appbin=out/'appbin';libbin=out/'libbin';appbin.mkdir();libbin.mkdir();cases=[writecase(n,out,source)for n in [80,160]]
 shellenv='source "'+str(of/'etc/bashrc')+'" >/dev/null 2>&1; export FOAM_USER_APPBIN="'+str(appbin)+'"; export FOAM_USER_LIBBIN="'+str(libbin)+'"; export OMP_NUM_THREADS=1; '
 commands=[]
 def run(name,command):
  cmd=['bash','-c',shellenv+command];start=time.time()
  with(logs/(name+'.log')).open('w')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1500)
  commands.append({'argv':cmd,'exit':r.returncode,'elapsed_seconds':time.time()-start});(out/'commands.json').write_text(json.dumps(commands,indent=2));assert r.returncode==0,(name,r.returncode)
 if not a.prepare_only:
  for case in cases:
   for tool in ['blockMesh','checkMesh','rhoSimpleFoam']:run(case.name+'-'+tool,'exec '+tool+' -case "'+str(case)+'"')
  src=out/'rhoSimpleWallAuditFoam';shutil.copytree(of/'applications/solvers/compressible/rhoSimpleFoam',src)
  (src/'Make/files').write_text('rhoSimpleFoam.C\n\nEXE = $(FOAM_USER_APPBIN)/rhoSimpleWallAuditFoam\n')
  e=src/'EEqn.H';s=e.read_text();assert s.count('+ thermophysicalTransport->divq(he)')==1;e.write_text(s.replace('+ thermophysicalTransport->divq(he)','+ thermophysicalTransport->divq(he)\n      + fvc::div(turbulence->devTau() & U)'))
  run('viscous-work-build','wmake "'+str(src)+'"')
  for old in cases:
   n=int(old.name.rsplit('-',1)[1]);case=out/('of10-reference-work-'+str(n));case.mkdir()
   for sub in ['0','constant','system']:shutil.copytree(old/sub,case/sub)
   c=case/'system/controlDict';c.write_text(c.read_text().replace('application rhoSimpleFoam','application rhoSimpleWallAuditFoam'))
   f=case/'system/fvSchemes';f.write_text(f.read_text().replace(' default none;',' default none;\n div((devTau&U)) Gauss linear;'))
   run(case.name,'exec "'+str(appbin/'rhoSimpleWallAuditFoam')+'" -case "'+str(case)+'"')
  for case in out.glob('of10-reference*'):
   tool=str(appbin/'rhoSimpleWallAuditFoam')if '-work-'in case.name else'rhoSimpleFoam'
   run(case.name+'-final-post','exec '+tool+' -case "'+str(case)+'" -postProcess -latestTime -funcs "(wallShearStress yPlus)"')
   for field in ['U','T','p','k','omega','nut']:
    # Full fields are local outputs, not automatically sent anywhere.
    assert any(case.glob('[1-9]*/'+field)),(case,field)
 files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()for p in out.rglob('*')if p.is_file()and'polyMesh'not in p.parts and'Make'not in p.parts}
 (out/'file-hashes.json').write_text(json.dumps(files,indent=2));print('Completed', 'input preparation'if a.prepare_only else 'OF10 independent references',out)

if __name__=='__main__':main()
