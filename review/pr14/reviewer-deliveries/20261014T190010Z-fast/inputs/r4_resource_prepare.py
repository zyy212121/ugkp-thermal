from r4_runner import *
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nS406实际SST22/冷壁资源恢复增量：canonical DeviceState/storage接口已变，旧owner22证据不能全复用。由固定当前源重新提取完整canonicalprefix/签名，链接本轮原O3的真实CHT/FSH nativearchive；保留原每22真实CUDA allocation错误/retry、实际copy错误、既有ownedpointer保持/reconfigure/destruction与mixed冷壁初始化copy失败。外部fault调用真实cudaMalloc过大请求/cudaMemcpy非法kind，不mock返回值或CUDA算术。追加safe物理资源观察不新增生产门禁；free/sync/hostallocation/concurrency未覆盖不能由普通cleanup泛化。\n')
s=(P/'r2_thin_build.py').read_text().replace('from r2_runner import *','from r4_runner import *\nimport pathlib').replace("P=pathlib.Path(__file__).parent","P=Path(__file__).resolve().parent")
s=s.replace('[12,17,18,23,26,28,33],900,','[12,17,18,23,26,28,33],')
s=s.replace("run('BUILD-REAL-CUDA-FAULT-OBSERVER'","run('BUILD-S406-REAL-CUDA-FAULT-OBSERVER'")
s=s.replace("run('BUILD-THIN-REAL-OWNER-'+app","run('BUILD-S406-THIN-REAL-OWNER-'+app")
s=s.replace("run('COLD-ACTUAL-COPY-RETRY'","run('S406-COLD-ACTUAL-COPY-RETRY'")
# Exact same runner template and real fault source, fresh final native libraries.
p=P/'r4_resource_build_generated.py';p.write_text(s)
shutil.copy2(p,D/'inputs'/p.name);shutil.copy2(P/'r2_thin_owner.cpp.in',D/'inputs/S406-original-thin-owner.cpp.in')
import subprocess
q=subprocess.run(['python3',str(p)]);assert q.returncode==0
for app in ['CHT','FSH']:
 b=ROOT/'work/fixtures'/('thin-real-owner-'+app);lib=ROOT/'work/cuda-actual-fault-observer.so'
 q=run('S406-SST22-ACTUAL-'+app,cmd('env','LD_PRELOAD='+str(lib),b,'sst'),[4,12,17,18,26,28],expected='Actualcurrentcanonical22SSTowners/nativeinitializer, all22 actualOOM/retry, realcopyerror/retry, repeatpointer/data anddestruction; nofullphysics/generalresourceclaim.',inputs=[D/'inputs'/('thin-real-owner-'+app+'.cu'),D/'inputs/cuda-actual-fault-observer.cpp'],binary=b,variant='DIAGNOSTIC_REAL_RESOURCE_S406',kind='RESOURCE_GPU')
 raw=(D/q['log']).read_text();assertion(q,q['exit_code']==0 and raw.count('PASS SST_ACTUAL_ALLOC_RETRY')==22 and 'PASS actualGPU SST22' in raw,'Actual22realallocationfailures+retry/copyretry/reconfigure/nativeKomegavalues/destruction; unrelatedfree/sync/hostfailure/concurrency notcovered.')
