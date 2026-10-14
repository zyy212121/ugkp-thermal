import re,shutil,json
from r2_runner import *
P=pathlib.Path(__file__).parent
fault=D/'inputs/cuda-actual-fault-observer.cpp';shutil.copy2(P/'r2_cuda_fault.cpp',fault);lib=ROOT/'work/cuda-actual-fault-observer.so'
if not lib.exists():
 t=run('BUILD-REAL-CUDA-FAULT-OBSERVER',cmd('g++','-std=c++17','-O2','-shared','-fPIC','-I/usr/local/cuda/include',fault,'-ldl','-o',lib),[12,26,28],expected='Calls realCUDA runtime with bounded huge allocation or invalidcopykind; no fake returncodes or CUDA mock.',inputs=[fault],kind='BUILD',variant='diagnostic-real-resource')
 assert not t['exit_code']
for app,sub,nativeLib in [('CHT','gpu','gpu/libugkpcht_cuda_internal.a'),('FSH','private_backend','private_backend/libugkwp_cuda_backend.a')]:
 nativePath=ROOT/f'applications/{app}/{sub}/GpuResidentStrict.cu';native=nativePath.read_text();start=native.index('struct DeviceState');end=native.index('\n};',start)+4;prefix=native[:end]
 def incl(m):
  for base in [nativePath.parent,ROOT/'common',ROOT/f'applications/{app}/gpu']:
   p=(base/m[1]).resolve()
   if p.exists():return '#include "'+str(p)+'"'
  return m[0]
 prefix=re.sub(r'#include "([^"]+)"',incl,prefix)+'\n}\n';header=D/'inputs'/('actual-'+app+'-state-prefix.cuh');header.write_text(prefix)
 names=['ugkwpGpuResidentStrictConfigureSst','ugkwpGpuResidentStrictRelease']+(['ugkwpGpuResidentStrictConfigureParticleStuckModel','ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch'] if app=='CHT' else [])
 declarations=[]
 for name in names:
  match=re.search(r'extern "C"\s+[^{};]*?\b'+name+r'\s*\([^{}]*?\)\s*\{',native);assert match,name;declarations.append(match[0].rsplit('{',1)[0]+';')
 match=re.search(r'int prepareSstOwners\([^)]*\)\s*\{([\s\S]*?)\n\}',native);assert match
 fields=re.findall(r'acquire\(s->(\w+),',match[1]);assert len(fields)==22
 source=D/'inputs'/('thin-real-owner-'+app+'.cu');source.write_text((P/'r2_thin_owner.cpp.in').read_text().replace('STATE_PREFIX',str(header)).replace('PROTOTYPES','\n'.join(declarations)).replace('SST_OWNER_FIELDS',','.join('s->'+x for x in fields)))
 binary=ROOT/'work/fixtures'/('thin-real-owner-'+app);archive=ROOT/f'applications/{app}'/nativeLib
 t=run('BUILD-THIN-REAL-OWNER-'+app,cmd('nvcc','-std=c++17','-O2','-arch=sm_89','--cudart=shared','-DCHT_TEST='+str(int(app=='CHT')),'-I'+str(nativePath.parent),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),'-I'+str(ROOT/f'applications/{app}/gpu'),source,archive,'-ldl','-o',binary),[12,17,18,23,26,28,33],900,'Thinactualcanonical struct + ORIGINAL compiledfullproduction library, actualexportednativeCUDA owners/kernels. No reimplementedGPU source.',inputs=[source,header,archive],kind='BUILD',variant='diagnostic-real-resource')
 if t['exit_code']:continue
 if app=='CHT':
  t=run('COLD-ACTUAL-COPY-RETRY',cmd('env','LD_PRELOAD='+str(lib),binary,'cold'),[12,23,26,28,33],expected='ActualCUDA runtimecopy error atperface initial H2D, pendingbit retained; actualcorrect retry/mixedfaces/coupleddefaultoverwrite priority; originalnativefunctions.',inputs=[source,header,fault,archive],binary=binary,variant='diagnostic-real-resource',kind='DIAGNOSTIC')
  raw=(D/t['log']).read_text();assertion(t,'PASS actualCUDA cold initialization' in raw and 'REAL_CUDA_COPY_ERROR code=0' not in raw,raw)
 print('READY SST actualthin',app,binary)
