"""Execute exact native admission bodies before their first CUDA operation."""
from pathlib import Path
import subprocess
import re
ROOT=Path(__file__).resolve().parents[3]

def test_actual_create_uses_finite_parameters_without_numerical_certificate(tmp_path):
    text=(ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    entry=text.split('extern "C" int ugkwpGpuResidentStrictCreate\n',1)[1]
    signature,body=entry.split(')\n{',1)
    body=body.split('    cudaError_t err = cudaFree(nullptr);',1)[0]
    parameters=[p.strip() for p in signature.lstrip('(').split(',')]
    names=[p.split()[-1] for p in parameters]
    args={n:'1' for n in names}
    args.update(handle='&handle',nCells='2',nFaces='3',nInternalFaces='1',nCellPlanes='4',particleCapacity='0',gasFluxScheme='2',gasReconstruction='0',gasLimiter='0',gasTimeIntegrator='1',turbulenceModel='0',particleBlockThreads='128',reductionBlockThreads='128',csrHeavyReductionMode='0',dragModel='0',gasRobustFallback='1',jammingPressureEnabled='0',gasMu='-3',gasPr='-.7',gammaGas='9',Rgas='-287',maxDiffusionNumber='-1',lesDeltaCoeff='-2',turbulentPrandtl='-.9',waleCw='-1',smagorinskyCs='-1')
    call=lambda:'ugkwpGpuResidentStrictCreate('+','.join(args[n] for n in names)+')'
    main='int main(){void*handle=nullptr;if('+call()+')return 1;'
    args['gasFluxScheme']='99';main+='if('+call()+'==0)return 2;}'
    src=tmp_path/'configuration.cpp';src.write_text('#include <cmath>\nvoid setLastErrorText(const char*){}\nextern "C" int ugkwpGpuResidentStrictCreate\n'+signature+')\n{'+body+'return 0;}\n'+main)
    exe=tmp_path/'configuration';b=subprocess.run(['g++','-std=c++17',str(src),'-o',str(exe)],capture_output=True,text=True);assert b.returncode==0,b.stderr
    assert subprocess.run([str(exe)]).returncode==0


def test_combined_dispatch_distinguishes_selected_particles_from_capacity(tmp_path):
    text=(ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    body=text.split('extern "C" int ugkwpGpuResidentStrictAdvance\n',1)[1].split(')\n{',1)[1].split('    if(checkGasBudgetAdvanceScope',1)[0]
    code=r'''
#include <cassert>
#include <string>
#include "common/gasTransport/GasModel.H"
struct DeviceState{struct{ugkwp::GasMode mode=ugkwp::GasMode::MixtureFrozen;}gasSpecies;bool particlesMayBePresent=false;int particleCapacity=100;};
DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}int validateState(DeviceState*,const char*){return 0;}
std::string error;void setLastErrorText(const char*p){error=p;}int gasCalls=0;
int ugkwpGpuResidentStrictAdvanceGasOnly(void*,double dt,double time){assert(dt==.1&&time==3);++gasCalls;return 0;}
int combined(void*handle,double dt,double simulationTime){BODY return 0;}
int main(){DeviceState s;assert(combined(&s,.1,3)==0&&gasCalls==1);s.particlesMayBePresent=true;assert(combined(&s,.1,3)!=0&&gasCalls==1);assert(error.find("no implementation")!=std::string::npos);}
'''.replace('BODY',body)
    source=tmp_path/'routing.cpp';source.write_text(code);binary=tmp_path/'routing'
    b=subprocess.run(['g++','-std=c++17','-I'+str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True);assert b.returncode==0,b.stderr
    assert subprocess.run([str(binary)]).returncode==0
