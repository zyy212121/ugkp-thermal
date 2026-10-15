from pathlib import Path as _ValidationPath
import sys as _validation_sys
_validation_sys.path.insert(0, str(next(p.parent for p in _ValidationPath(__file__).resolve().parents if p.name == "validation")))
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import os,re,resource,subprocess,sys
root,out=Path(sys.argv[1]),Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
app=sys.argv[3] if len(sys.argv)>3 else 'gasUGKP'
source=root/'applications'/app/('gpu' if app=='CHT' else 'private_backend')
text=(source/'GpuResidentStrict.cu').read_text();a=text.index('struct DeviceState');b=text.index('\n};',a)
state=text[a:b].replace('#include "GpuAutomaticCsrScheduleFields.inl"',(root/'common/GpuAutomaticCsrScheduleFields.inl').read_text())
fields=re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',state,re.M)
fields=[(ty,n) for ty,n in fields if n not in ['diagnosticPreTransportParticleCount','sourceInjectedCount'] and not n.startswith('flatPressure')]
limits={'csrMultiTaskCellList':4,'csrHeavyCellCount':1,'csrHeavyTaskCount':1,'csrHeavyTaskCursor':1,'csrMaximumOccupancy':1,'csrHeavyPartials':8*128}
code=(VALIDATION_ROOT / 'tests/fixtures/gas_auto/gas_auto_pool.cu.in').read_text().replace('ALLOCATE_FIELDS','\n'.join(f'mem(s->{n},{limits.get(n,4096)});' for _,n in fields))
production_chain=len(sys.argv)>4 and sys.argv[4]=='production-directory'
if production_chain:
    code=code[:code.index('int main(')]+(VALIDATION_ROOT / 'tests/fixtures/gas_auto/production_directory.cu.in').read_text()
    def stage(name):
        begin=text.index('    UGKP_DEV_PROBE_ENTER('+name+');',text.index('    UGKP_DEV_PROBE_ENTER(ProbeBinPre);'))
        match=re.search(r'    UGKP_DEV_PROBE_LEAVE(?:_IF)?\('+name+r'[^;]*;',text[begin:])
        assert match,name
        end=begin+match.end()
        return text[begin:end]
    code=code.replace('PRODUCTION_PRE_STAGE',stage('ProbeBinPre')).replace('PRODUCTION_COMPACTION_STAGE',stage('ProbeCompaction'))
prefix=''
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
'''.replace('TOLERANCE','2e-12')
code=code.replace('TEST_ADAPTERS',prefix+adapter).replace('TEST_LIGHT_BLOCKS','lightBlocksPerSm' if app=='gasUGKP' else 'lightResidentBlocksPerSm')
cu=out/'gas_auto_pool.cu';cu.write_text(code);exe=out/'gas_auto_pool'
resource.setrlimit(resource.RLIMIT_STACK,(512*1024*1024,resource.RLIM_INFINITY));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'--fmad='+('false' if app=='CHT' else 'true'),'-I'+str(source),'-I'+str(root/'common'),'-I'+str(root/'applications'/app/'gpu'),str(cu),'-o',str(exe)]
if app=='CHT':cmd.append(str(source/'GpuWallEnergy64.cu'))
if production_chain:cmd+=['-Xlinker','--wrap=cudaLaunchKernel','-Xlinker','--wrap=__cudaLaunchKernel']
with (out/'build.log').open('w') as f:q=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
assert q.returncode==0,(out/'build.log').read_text()[-6000:]
print(exe)
