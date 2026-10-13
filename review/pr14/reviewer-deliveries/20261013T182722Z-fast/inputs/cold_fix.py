import pathlib,shutil,json,difflib
from run_fast import ROOT,D,run,cmd,amend,sha
overlay=ROOT/'work/diagnostic-P007'
for p in ['common','applications/CHT']:
 shutil.copytree(ROOT/p,overlay/p,ignore=shutil.ignore_patterns('linux64*','docs','*.a','__pycache__'))
p=overlay/'applications/CHT/gpu/GpuResidentStrict.cu';old=p.read_text()
needle='''    const cudaError_t error = cudaMemset
    (s->particleWallEffusivityByFace, 0, nFaces*sizeof(GpuReal));'''
replacement='''    // Seed the existing uncoupled-wall scalar material fallback. The host
    // coupler subsequently overwrites its actual selected wall patch faces.
    const GpuReal fallbackEffusivity = ::sqrt
    (s->particleWallDensityKgM3*s->particleWallSpecificHeatJkgK
     *s->particleWallConductivityWmK);
    const std::vector<GpuReal> initialEffusivity(nFaces, fallbackEffusivity);
    const cudaError_t error = cudaMemcpy
    (s->particleWallEffusivityByFace, initialEffusivity.data(),
     nFaces*sizeof(GpuReal), cudaMemcpyHostToDevice);'''
assert old.count(needle)==1
new=old.replace(needle,replacement);p.write_text(new)
patch=D/'patches/P007-cold-wall-fallback-producer.patch';patch.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/applications/CHT/gpu/GpuResidentStrict.cu',tofile='b/applications/CHT/gpu/GpuResidentStrict.cu')))
(D/'patches/P007-cold-wall-fallback-producer.json').write_text(json.dumps({'classification':'isolated minimal source correction, diagnostic only','source_before_sha256':sha(ROOT/'applications/CHT/gpu/GpuResidentStrict.cu'),'source_after_sha256':sha(p),'patch_sha256':sha(patch),'reason':'P006 observes allocated per-face effusivity zero, scalar fallback11304.866..., age0 wallG NaN. Existing producer zero-init hid scalar fallback on uncoupled candidate faces.','scope':'Seed from existing DeviceState scalar fallback, retain coupled patch override and pending flags. No finite/positive check, clamp, rejection or new solver gate.','limitations':['Default device material1800/710/100; user example dictionary7900/500/15 has no host consumer, not claimed fixed.','Reconfiguration/scalar changing and coupled patches require developer formal audit/recheck.'],'formal_fix':False,'production_checkout_untouched':True},indent=2)+'\n')
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n追加P007：已先用原版完整field断言复现cold2D NaN（red），P006确认零face effusivity的producer链。仅隔离初始化producer使用现有DeviceState scalar fallback填充，后续真实coupled patch覆盖不动；不加数值门禁，完全相同INPUT2，不改物性输入/容差。原10步1D/2D重跑、full profile/carry/temp/fields有限性及publication检查，300s/例，900s编译；该通过属于补丁版本。\n')
binary=ROOT/'work/install/P007/bin/CHT';binary.parent.mkdir(parents=True,exist_ok=True)
t=run('BUILD-CHT-P007',cmd('env',f'FOAM_USER_APPBIN={binary.parent}','bash',overlay/'applications/CHT/tools/build_cuda_solver.sh'),[12,26,33],900,'Same native CHT flags; seed existing fallback producer, no numerical gate.',inputs=[patch],kind='BUILD',variant='diagnostic-P007')
if t['exit_code']:raise SystemExit(1)
for model in ['1D','2D']:
 case=ROOT/'work/cases/resident'/('CHT-coldWall'+model+'-P007');src=D/'inputs/native/resident'/('CHT-coldWall'+model+'-INPUT2');shutil.copytree(src,case)
 t=run('MESH-COLD'+model+'-P007',cmd('blockMesh','-case',case),[33],300,'same input fresh diagnostic mesh',inputs=list(src.rglob('*')),kind='SETUP',variant='diagnostic-P007')
 if t['exit_code']:continue
 t=run('CHT-COLD'+model+'-P007',cmd(binary,'-case',case),[12,26,33],300,'All actual 8/64 profile nodes and double carry/temperature finite; particle/ledger state decoded separately; no original PASS promotion.',inputs=list(src.rglob('*')),binary=binary,variant='diagnostic-P007')
 out=D/'artifacts/resident'/case.name;out.mkdir(parents=True,exist_ok=True)
 for q in case.iterdir():
  try:v=float(q.name)
  except ValueError:continue
  if v>0:shutil.copytree(q,out/q.name)
 amend(t['id'],patch_id='P007',artifacts=[str(out.relative_to(D))])
