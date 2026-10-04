from pathlib import Path
import os,re,resource,subprocess,sys
root,out=Path(sys.argv[1]),Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
app=sys.argv[3] if len(sys.argv)>3 else 'gasUGKP'
bits=int(sys.argv[4]) if len(sys.argv)>4 else 64
source=root/'applications'/app/('gpu' if app=='CHT' else 'private_backend')
text=(source/'GpuResidentStrict.cu').read_text();a=text.index('struct DeviceState');b=text.index('\n};',a)
state=text[a:b].replace('#include "GpuAutomaticCsrScheduleFields.inl"',(root/'common/GpuAutomaticCsrScheduleFields.inl').read_text())
fields=re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',state,re.M)
fields=[(ty,n) for ty,n in fields if n not in ['diagnosticPreTransportParticleCount','sourceInjectedCount'] and (bits==32 or not n.startswith('flatPressure'))]
limits={'csrMultiTaskCellList':4,'csrHeavyCellCount':1,'csrHeavyTaskCount':1,'csrHeavyTaskCursor':1,'csrMaximumOccupancy':1,'csrHeavyPartials':8*128}
code=Path(__file__).with_name('gas_auto_pool.cu.in').read_text().replace('ALLOCATE_FIELDS','\n'.join(f'mem(s->{n},{limits.get(n,4096)});' for _,n in fields))
prefix='using namespace ugkwpCudaFp32;\n' if app=='CHT' and bits==32 else ''
if app=='gasUGKP':
    adapter='''using TestDirectory=HeavyDirectoryKind;
void selectDirectory(DeviceState*s,TestDirectory k){s->useSplitPreDirectory=k!=TestDirectory::full;s->preInjectionSegmentActive=k==TestDirectory::splitBaseAndInjection;}
int prepareTasks(DeviceState*s,int b,TestDirectory k){return prepareCsrSegmentedReductionTasks(s,b,k);}
int decide(DeviceState*s,int b,TestDirectory k){return runToolB3(s,b,k);}
void clearPool(DeviceState*s,double dt){clearPoissonThermalPoolKernel<<<1,32>>>(s->deviceState,dt);}
int heavyPool(DeviceState*s,double dt,int b){return launchCsrHeavyPoolReduction(s,dt,true,b,s->useSplitPreDirectory);}
int lightPool(DeviceState*s,TestDirectory k,double dt,int b){if(k==TestDirectory::full){accumulatePoissonPoolParticlesByCellKernel<false><<<4,b,8*b*sizeof(double)>>>(s->deviceState,dt);return 0;}return launchSplitPrePoissonPoolLightReduction(s,dt,b,8*b*sizeof(double));}
int expectedSource(TestDirectory k){return int(k==TestDirectory::full?CsrReductionTaskSource::fullIndexed:(k==TestDirectory::baseOnly?CsrReductionTaskSource::splitBaseDirect:CsrReductionTaskSource::splitLogical));}
int directoryTag(TestDirectory k){return int(k);}
constexpr double tolerance=2e-12;
'''
else:
    adapter='''enum class TestDirectory {full,baseOnly,splitBaseAndInjection};
void selectDirectory(DeviceState*s,TestDirectory k){s->splitPreDirectoryActive=k!=TestDirectory::full;s->nBoundarySources=k==TestDirectory::splitBaseAndInjection?1:0;}
int prepareTasks(DeviceState*s,int b,TestDirectory k){if(updateDynamicHeavyPolicy(s)!=0)return 1;return prepareCsrSegmentedReductionTasks(s,b,k!=TestDirectory::full);}
int decide(DeviceState*s,int b,TestDirectory){return runToolB3(s,b);}
void clearPool(DeviceState*s,double){clearPoissonThermalPoolKernel<<<1,32>>>(s->deviceState);}
int heavyPool(DeviceState*s,double dt,int b){return launchCsrHeavyPoolReduction(s,GpuTime(dt),true,b);}
int lightPool(DeviceState*s,TestDirectory k,double dt,int b){if(k==TestDirectory::full)accumulatePoissonPoolParticlesByCellKernel<<<4,b,8*b*sizeof(GpuReal)>>>(s->deviceState,GpuTime(dt));else{accumulatePoissonPoolSplitSegmentKernel<true,false><<<4,b,8*b*sizeof(GpuReal)>>>(s->deviceState,GpuTime(dt));if(k==TestDirectory::splitBaseAndInjection)accumulatePoissonPoolSplitSegmentKernel<false,true><<<4,b,8*b*sizeof(GpuReal)>>>(s->deviceState,GpuTime(dt));}return int(cudaGetLastError()!=cudaSuccess);}
int expectedSource(TestDirectory k){return int(k==TestDirectory::full?CsrReductionTaskSource::fullIndexed:CsrReductionTaskSource::splitLogical);}
int directoryTag(TestDirectory k){return int(k!=TestDirectory::full);}
constexpr double tolerance=TOLERANCE;
'''.replace('TOLERANCE','2e-5' if bits==32 else '2e-12')
code=code.replace('TEST_ADAPTERS',prefix+adapter).replace('TEST_LIGHT_BLOCKS','lightBlocksPerSm' if app=='gasUGKP' else 'lightResidentBlocksPerSm')
cu=out/'gas_auto_pool.cu';cu.write_text(code);exe=out/'gas_auto_pool'
resource.setrlimit(resource.RLIMIT_STACK,(512*1024*1024,resource.RLIM_INFINITY));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'--fmad='+('false' if app=='CHT' else 'true'),'-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(source),'-I'+str(root/'common'),'-I'+str(root/'applications'/app/'gpu'),str(cu),'-o',str(exe)]
if app=='CHT':cmd.append(str(source/'GpuWallEnergy64.cu'))
with (out/'build.log').open('w') as f:q=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
assert q.returncode==0,(out/'build.log').read_text()[-6000:]
print(exe)
