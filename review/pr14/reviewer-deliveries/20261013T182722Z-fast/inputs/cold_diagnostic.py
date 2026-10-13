import pathlib,shutil,json,difflib
from run_fast import ROOT,D,run,cmd,amend,sha
overlay=ROOT/'work/diagnostic-P006'
for p in ['common','applications/CHT']:
 shutil.copytree(ROOT/p,overlay/p,ignore=shutil.ignore_patterns('linux64*','docs','*.a','__pycache__'))
p=overlay/'common/wall/GpuColdWall2DDevice.cuh';old=p.read_text();new=old
new=new.replace('        const ColdWallThermalReal localGasConductanceWK =','        if(particleI==0 && radialNode==0 && substep==0) printf("PR14_DIAG age=%.17g eff=%.17g wallG=%.17g wetArea=%.17g wallT=%.17g gasG=%.17g oldCarry=%.17g\\n",double(localAge),double(wallEffusivity),double(wallConductanceWK),double(wetAreaM2),double(wallTemperatureK),double(gasConductanceWK),double(oldCarry));\n        const ColdWallThermalReal localGasConductanceWK =')
new=new.replace('            maximumAreaM2 = static_cast<GpuReal>', '            if(particleI==0) printf("PR14_DIAG face=%d effArray=%.17g effScalar=%.17g rho=%.17g Cp=%.17g k=%.17g\\n",wallFaceI,double(wallEffusivity),sqrt(double(s.particleWallDensityKgM3)*s.particleWallSpecificHeatJkgK*s.particleWallConductivityWmK),double(s.particleWallDensityKgM3),double(s.particleWallSpecificHeatJkgK),double(s.particleWallConductivityWmK));\n            maximumAreaM2 = static_cast<GpuReal>')
# Source indentation is twelve spaces at this point, verify the insertion.
if 'PR14_DIAG face=' not in new:
 new=new.replace('            maximumAreaM2', '            maximumAreaM2')
 new=new.replace('            wallEffusivity =', '            wallEffusivity =')
 new=new.replace('            maximumAreaM2 =', '            maximumAreaM2 =')
if 'PR14_DIAG face=' not in new:
 needle='            maximumAreaM2 = static_cast<GpuReal>'
 assert needle not in old
 needle='            maximumAreaM2'
 # Actual file has 12 spaces? detect the line instead of changing semantics.
 line=next(s for s in old.splitlines() if 'maximumAreaM2 = static_cast<GpuReal>' in s)
 new=new.replace(line,'            if(particleI==0) printf("PR14_DIAG face=%d effArray=%.17g effScalar=%.17g rho=%.17g Cp=%.17g k=%.17g\\n",wallFaceI,double(wallEffusivity),sqrt(double(s.particleWallDensityKgM3)*s.particleWallSpecificHeatJkgK*s.particleWallConductivityWmK),double(s.particleWallDensityKgM3),double(s.particleWallSpecificHeatJkgK),double(s.particleWallConductivityWmK));\n'+line)
assert new!=old and 'PR14_DIAG face=' in new and 'PR14_DIAG age=' in new
p.write_text(new)
patch=D/'patches/P006-cold2D-printf-diagnostic.patch';patch.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/common/wall/GpuColdWall2DDevice.cuh',tofile='b/common/wall/GpuColdWall2DDevice.cuh')))
(D/'patches/P006-cold2D-printf-diagnostic.json').write_text(json.dumps({'classification':'isolated diagnostic only','production_before_sha256':sha(ROOT/'common/wall/GpuColdWall2DDevice.cuh'),'production_after_sha256':sha(p),'physical_equations_modified':False,'hypothesis':'nonempty per-face effusivity backing remains zero on uncoupled candidate wall, first age0 gives sqrt(0)/0 and poisons paired mean correction','source_checkout_untouched':True},indent=2)+'\n')
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n追加P006 cold2D首次NaN诊断：完全相同INPUT2、O3/sm89/fmad=false，隔离源仅device printf读取已有计算值，记录face/scalar effusivity、age、wallG、carry。300s native，900s build；原NaN日志和所有输出保留，printf版本不用于速度比较。\n')
binary=ROOT/'work/install/P006/bin/CHT';binary.parent.mkdir(parents=True,exist_ok=True)
t=run('BUILD-CHT-P006',cmd('env',f'FOAM_USER_APPBIN={binary.parent}','bash',overlay/'applications/CHT/tools/build_cuda_solver.sh'),[12,26,33],900,'Same native CHT flags, only isolated device printf diagnosis.',inputs=[patch],kind='BUILD',variant='diagnostic-P006')
if t['exit_code']:raise SystemExit(1)
case=ROOT/'work/cases/resident/CHT-coldWall2D-P006';src=D/'inputs/native/resident/CHT-coldWall2D-INPUT2';shutil.copytree(src,case)
t=run('MESH-COLD2D-P006',cmd('blockMesh','-case',case),[33],300,'same input fresh diagnostic mesh',inputs=list(src.rglob('*')),kind='SETUP',variant='diagnostic-P006')
if t['exit_code']:raise SystemExit(1)
t=run('CHT-COLD2D-P006',cmd(binary,'-case',case),[12,26,33],300,'Locate first NaN without formula changes. Original result is not promoted by diagnostic execution.',inputs=list(src.rglob('*')),binary=binary,variant='diagnostic-P006')
raw=(D/t['log']).read_text();lines=[s for s in raw.splitlines() if 'PR14_DIAG' in s]
print('\n'.join(lines));amend(t['id'],verdict='FAIL',observations='Diagnostic-only execution; inspect first NaN cause: '+'\n'.join(lines),patch_id='P006')
out=D/'artifacts/resident/CHT-coldWall2D-P006';out.mkdir(parents=True,exist_ok=True)
for q in case.iterdir():
 try:v=float(q.name)
 except ValueError:continue
 if v>0:shutil.copytree(q,out/q.name)
