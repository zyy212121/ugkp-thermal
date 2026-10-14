from r3_runner import *
import difflib
overlay=ROOT/'work/owner-entry-S305';assert not overlay.exists()
for folder in ['common','applications/gasUGKP/private_backend','applications/gasUGKP/gpu']:
 shutil.copytree(ROOT/folder,overlay/folder)
p=overlay/'common/GpuGasAdvance.cuh';old=p.read_text();anchor='    ugkwp::invalidateGasBoundaryLayerPreparation(*s);\n    if (TrialPolicy::begin(s) != 0)'
assert old.count(anchor)==1
new=old.replace(anchor,'    std::fprintf(stdout,"R3CALL dt=%a time=%a native=%d\\n",double(dt),double(simulationTime),int(nativeRefinement));\n'+anchor)
p.write_text(new)
patch=D/'patches/S305-readonly-native-trial-entry.patch';patch.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/common/GpuGasAdvance.cuh',tofile='b/isolated/common/GpuGasAdvance.cuh')))
(D/'inputs/GpuGasAdvance-owner-entry-S305.cuh').write_text(new)
source=D/'inputs/chemistry-owner-chronology-S304.cu';binary=ROOT/'work/fixtures/chemistry-owner-entry-S305'
command=json.loads((ROOT/'work/fixtures/chemistry-trial-formal/command.json').read_text())
command=[a.replace(str(ROOT),str(overlay)) if a.startswith('-I') else a for a in command];command[-3]=str(source);command[-1]=str(binary)
(D/'inputs/chemistry-owner-entry-S305-command.json').write_text(json.dumps(command,indent=2)+'\n')
(D/'metadata/owner-entry-overlay-hashes.json').write_text(json.dumps({'files':{str(q.relative_to(overlay)):sha(q) for q in overlay.rglob('*') if q.is_file()},'only_algorithm_file_modified':'common/GpuGasAdvance.cuh','modification':'fprintf at actual advanceGasTrial entry; no arithmetic/CUDA/state change'},indent=2)+'\n')
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nS305补足S304观察缺口：S304 currentDuration仅在gravity applySources更新，早期chemistry/Co拒绝前未更新，因此拒绝dt打印0或旧accepted dt，原日志保留为部分chronology，不当attempt duration证明。新隔离copy common/private_backend/gpu，只在真实advanceGasTrial入口fprintf实际dt/time/native；正式源不改，使用同四原owner成功控制、全部原assertion，16失败矩阵不重跑。输出去除IO行应与正式owner stdout bitexact。compile900/run300，正常cleanup，I/O不参与性能。\n')
t=run('BUILD-CHEMISTRY-OWNER-ENTRY-S305',cmd(*command),[10,12,14,26],900,'Readonlyactualnative advanceGasTrial entry dt/time; no false/stale rejected duration from applySources hook.',inputs=[source,patch,p],kind='BUILD',variant='DIAGNOSTIC_S305');assert t['exit_code']==0
