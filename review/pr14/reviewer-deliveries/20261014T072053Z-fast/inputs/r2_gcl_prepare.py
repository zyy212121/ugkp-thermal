import difflib
from r2_runner import *
app=ROOT/'applications/CHMT';original=(app/'CHMT.C').read_text();s=original.replace('end,false,input.state,report,error','end,true,input.state,report,error');assert s!=original
anchor='            publish();\n            std::cout<<"CHMT accepted time="'
observer=r'''            for(const auto& micro:report.geometry)for(int stage=0;stage<2;++stage){
                const auto& q=micro.gas[stage];
                std::fprintf(stderr,"GCL2_META,%llu,%llu,%d,%a,%a,%a\n",static_cast<unsigned long long>(input.state.acceptedSteps),static_cast<unsigned long long>(micro.microSequence),stage,micro.begin,micro.end,q.interval);
                for(std::size_t i=0;i<q.oldVolume.size();++i)std::fprintf(stderr,"GCL2_CELL,%llu,%llu,%d,%zu,%a,%a,%a\n",static_cast<unsigned long long>(input.state.acceptedSteps),static_cast<unsigned long long>(micro.microSequence),stage,i,q.oldVolume[i],q.newVolume[i],q.evaluationVolume[i]);
                for(std::size_t i=0;i<q.sweptVolume.size();++i)std::fprintf(stderr,"GCL2_FACE,%llu,%llu,%d,%zu,%a\n",static_cast<unsigned long long>(input.state.acceptedSteps),static_cast<unsigned long long>(micro.microSequence),stage,i,q.sweptVolume[i]);
            }
'''
assert s.count(anchor)==1;s=s.replace(anchor,observer+anchor);s='#include <cstdio>\n'+s
source=D/'inputs/CHMT-retain-geometry-diagnostic.C';source.write_text(s);patch=D/'patches/D202-retain-actual-microgeometry.patch';patch.write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/applications/CHMT/CHMT.C',tofile='b/applications/CHMT/CHMT.C')))
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nD202动网格证据补充：正式frontend固定retainStageGeometry=false，仅checkpoint保留最后微步两stage，不能据此关闭全部GCL。隔离复制CHMT.C只启用已有gasWindowStageGeometry记录功能、打印原始old/new/evaluation/swept数组，实际GPU Backend.o仍为本轮正式原对象，完整物理源码/headers/flags未改。追加hostlink900s，moving-GCL诊断300s并逐bytes比较所有原输出；在原Ns10 movingchem input也执行同一host诊断以配对S206真实1400化学launch参数。原始profile/性能不使用诊断IO。\n')
for ns in [2,10]:
 build=ROOT/'work/build'/('CHMT'+str(ns));b=ROOT/'work/fixtures'/('CHMT-geometry-'+str(ns))
 flags=['-std=c++17','-O2','-DNoRepository','-DWM_DP','-DWM_LABEL_SIZE=32','-DUGKWP_GAS_SPECIES='+str(ns),'-I'+str(app),'-I'+str(ROOT/'common'),'-I'+str(build/'generated'),'-I/opt/openfoam10/src/OpenFOAM/lnInclude','-I/opt/openfoam10/src/finiteVolume/lnInclude','-I/opt/openfoam10/src/meshTools/lnInclude','-I/opt/openfoam10/src/OSspecific/POSIX/lnInclude']
 files=[source,*[app/x for x in ['io/MultirateEvolution.C','materials/CpuMaterialDriver.C','materials/MaterialTransport.C','film/CpuFilmDriver.C','ablation/CpuSurfaceInterface.C','mesh/Geometry.C','restart/Checkpoint.C']],build/'Backend.o']
 libs=['-L/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib','-lfiniteVolume','-lmeshTools','-lOpenFOAM','-L/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib/dummy','-lPstream','-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart']
 run('BUILD-ACTUAL-MICROGEOMETRY-'+str(ns),cmd('g++',*flags,*files,*libs,'-o',b),[8,10,14,26,29],900,'Actualunmodified GPU Backend.o +original physicalhostunits; observermain boolretention/IO only.',inputs=files,kind='BUILD',variant='diagnostic-geometry-D202')
