import pathlib,shutil,difflib,json,subprocess
from run_fast import ROOT,D,run,cmd,amend,sha
checker=D/'inputs/resident_check.py';old=checker.read_text();new=old.replace('float(nodes.mean())','float(nodes.astype(np.float64).mean())')
assert old!=new
checker2=D/'inputs/resident_check_fp64_mean.py';checker2.write_text(new)
(D/'patches/P009-checker-mean-double.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/reviewer/resident_check.py',tofile='b/reviewer/resident_check_fp64_mean.py')))
(D/'patches/P009-checker-mean-double.json').write_text(json.dumps({'classification':'test support only','production_modified':False,'reason':'Numpy mean of float profile default float32 accumulator introduced checker error; independent physical mean must promote stored nodes to double. No tolerance/assertion removed; old failed results retained.','old_checker_sha256':sha(checker),'corrected_checker_sha256':sha(checker2)},indent=2)+'\n')
for name in ['CHT-coldWall1D-INPUT2','CHT-coldWall2D-P007']:
 t=run('FIELDS2-RED-'+name,cmd('python3',checker2,'finite',name),[23,26,33],300,'Correct double-accumulated profile/carry inverse check abs1e-9 before production init-carry patch; preserve first cold activation mismatch.',inputs=[checker2],kind='ANALYSIS',variant='diagnostic-P007' if 'P007' in name else 'original')
 amend(t['id'],verdict='FAIL' if t['exit_code'] else 'PASS',test_support_patch='P009',observations=(D/t['log']).read_text())
 print(name,'first profile fail',json.loads((D/t['log']).read_text())['failures'])
overlay=ROOT/'work/diagnostic-P010';shutil.copytree(ROOT/'work/diagnostic-P007',overlay,ignore=shutil.ignore_patterns('linux64*','docs','*.a','__pycache__'))
patches=[];source=[]
for relative in ['common/operators/clearColdWallParticleState.cuh','common/wall/GpuColdWall2DDevice.cuh']:
 p=overlay/relative;before=p.read_text();after=before
 if 'clearCold' in relative:
  marker='    for (int ring = 0; ring < Foam::gpuThermal::coldWallRadialRingCount; ++ring)\n    {\n        s.pColdRingSolidMass[ringBase + ring] = 0.0;\n    }\n    s.pColdFrozenArea[particleI] = 0.0;\n    s.pColdContactAge[particleI] = 0.0;\n}'
  # Last occurrence is initialization; clear must remain unchanged.
  pos=before.rfind(marker);assert pos>=0
  addition='''#ifdef GPU_PAIRED_PARTICLE_HEAT
    // Preserve the preexisting physical enthalpy through the profile cache cast.
    s.pThermalEnthalpyResidual[particleI] += double(nodeEnthalpy[0])
      - double(s.pColdNodeSpecificEnthalpy[nodeBase]);
#endif
'''
  after=before[:pos]+addition+before[pos:]
 else:
  marker='    s.pCold2DFrozenArea[particleI] = 0.0;\n}'
  pos=before.find(marker,before.find('__device__ void initialiseColdWall2DParticleState'));assert pos>=0
  addition='''#ifdef GPU_PAIRED_PARTICLE_HEAT
    // Preserve the preexisting physical enthalpy through the profile cache cast.
    s.pThermalEnthalpyResidual[particleI] += double(initialEnthalpy)
      - double(s.pCold2DNodeSpecificEnthalpy[nodeBase]);
#endif
'''
  after=before[:pos]+addition+before[pos:]
 assert after!=before;p.write_text(after);patches.append(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative)));source.append({'path':relative,'before_sha256':sha(ROOT/relative),'after_sha256':sha(p)})
patch=D/'patches/P010-cold-profile-initialization-carry.patch';patch.write_text(''.join(patches))
(D/'patches/P010-cold-profile-initialization-carry.json').write_text(json.dumps({'classification':'isolated source correction/diagnostic only','finding':'GPU-F004','requires':'P007 for uncoupled effusivity producer','sources':source,'reason':'At first wall contact, double enthalpy cast into uniform float nodes discarded quantization remainder; double carry still only holds precontact ordinary heat remainder. Corrected independent check first mismatch7.47809e-5K, before NaN.','scope':'Retain unchanged profile cast/formula and add exactly its quantization loss to existing double carry at the original producer. No numeric rejection/clamp/gate.','formal_fix':False,'limits':['CHT two models only; FSH shared template needs developer formal build/real GPU recheck.','Full wall/gas ledger balance and mixed coupled faces not claimed.'],'source_checkout_untouched':True},indent=2)+'\n')
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n追加P009/P010：保留旧checker mean32失败；仅将独立mean改用double，不改1e-9K容差或失败断言。FIELDS2-RED再次记录首次profile cast偏差后，P010隔离补上init producer所丢焓到既有double carry，原float cast及物理law不改，300s/原生例、900s build，复核全部写出time含首contact。P007 producer补丁合用，不能算原PR通过。\n')
binary=ROOT/'work/install/P010/bin/CHT';binary.parent.mkdir(parents=True,exist_ok=True)
t=run('BUILD-CHT-P010',cmd('env',f'FOAM_USER_APPBIN={binary.parent}','bash',overlay/'applications/CHT/tools/build_cuda_solver.sh'),[12,23,26,33],900,'Isolated P007+P010 original CHT O3/sm89/fmadfalse.',inputs=[patch,D/'patches/P007-cold-wall-fallback-producer.patch'],kind='BUILD',variant='diagnostic-P007+P010')
if t['exit_code']:raise SystemExit(1)
for model in ['1D','2D']:
 src=D/'inputs/native/resident'/('CHT-coldWall'+model+'-INPUT2');case=ROOT/'work/cases/resident'/('CHT-coldWall'+model+'-P010');shutil.copytree(src,case)
 t=run('MESH-COLD'+model+'-P010',cmd('blockMesh','-case',case),[33],300,'Identical physical input, unchanged tolerances.',inputs=list(src.rglob('*')),kind='SETUP',variant='diagnostic-P007+P010')
 if t['exit_code']:continue
 t=run('CHT-COLD'+model+'-P010',cmd(binary,'-case',case),[12,23,26,33],300,'Same input, all actual profile/carry/temperature and physical particles finite and independent caloric abs1e-9K at all written times incl first activation.',inputs=list(src.rglob('*')),binary=binary,variant='diagnostic-P007+P010')
 out=D/'artifacts/resident'/case.name;out.mkdir(parents=True,exist_ok=True)
 for q in case.iterdir():
  try:v=float(q.name)
  except ValueError:continue
  if v>0:shutil.copytree(q,out/q.name)
 check=run('FIELDS3-'+case.name,cmd('python3',checker2,'finite',case.name),[23,26,33],300,'Same corrected mean-double check, original thresholds, all fields/times.',inputs=[checker2],kind='ANALYSIS',variant='diagnostic-P007+P010')
 amend(check['id'],verdict='PASS' if check['exit_code']==0 else 'FAIL',observations=(D/check['log']).read_text());amend(t['id'],verdict='PASS' if check['exit_code']==0 else 'FAIL',checker_result_id=check['id'],patch_id='P007+P010')
