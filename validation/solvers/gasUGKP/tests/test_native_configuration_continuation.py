"""Execute exact native admission bodies before their first CUDA operation."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess

ROOT = SOURCE_ROOT


def prefix_before(text, entry, boundary):
    signature, body = text.split('extern "C" int ' + entry + '\n', 1)[1].split(')\n{', 1)
    assert body.count(boundary) == 1
    return signature, body.split(boundary, 1)[0]


def test_actual_create_distinguishes_required_handle_from_numeric_metadata(tmp_path):
    text = (ROOT / 'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    signature, body = prefix_before(
        text, 'ugkwpGpuResidentStrictCreate',
        '    cudaError_t err = cudaFree(nullptr);',
    )
    names = [parameter.strip().split()[-1] for parameter in signature.lstrip('(').split(',')]
    args = {name: '1' for name in names}
    args.update(
        handle='&handle', nCells='2', nFaces='3', nInternalFaces='1',
        nCellPlanes='4', particleCapacity='0', gasFluxScheme='2',
        gasReconstruction='0', gasLimiter='0', gasTimeIntegrator='1',
        turbulenceModel='0', particleBlockThreads='128', reductionBlockThreads='128',
        csrHeavyReductionMode='0', dragModel='0',
        jammingPressureEnabled='0', gasMu='-3', gasPr='-.7', gammaGas='9',
        Rgas='-287', lesDeltaCoeff='-2',
        turbulentPrandtl='-.9', waleCw='-1', smagorinskyCs='-1',
    )

    def call():
        return 'ugkwpGpuResidentStrictCreate(' + ','.join(args[name] for name in names) + ')'

    checks = []
    for updates in ({}, {'particleCapacity': '100', 'gasFluxScheme': '99'},
                    {'gasMu': 'std::nan("")', 'Rgas': 'INFINITY'}):
        args.update(updates)
        checks.append('handle=reinterpret_cast<void*>(1);assert(' + call()
                      + '==73&&handle==nullptr);')
    args['handle'] = 'nullptr'
    checks.append('assert(' + call() + '==1&&cudaBoundaries==3);')
    code = ('#include <cassert>\n#include <cmath>\n'
            'int cudaBoundaries=0;void setLastErrorText(const char*){}\n'
            'extern "C" int ugkwpGpuResidentStrictCreate\n' + signature + ')\n{'
            + body + '++cudaBoundaries;return 73;}\n'
            'int main(){void*handle=nullptr;' + ''.join(checks) + '}')
    source = tmp_path / 'configuration.cpp'
    source.write_text(code)
    binary = tmp_path / 'configuration'
    result = subprocess.run(['g++', '-std=c++17', str(source), '-o', str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert subprocess.run([str(binary)]).returncode == 0


def test_combined_dispatch_distinguishes_selected_particles_from_capacity(tmp_path):
    text = (ROOT / 'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    _, body = prefix_before(text, 'ugkwpGpuResidentStrictAdvance',
                            '    if (tuneFixedWorkBlockThreads(s, dt, simulationTime) != 0)')
    code = r'''
#include <cassert>
#include <string>
#include "common/gasTransport/GasModel.H"
struct DeviceState{struct{ugkwp::GasMode mode=ugkwp::GasMode::MixtureFrozen;}gasSpecies;bool particlesMayBePresent=false;int particleCapacity=100;};
DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}
int validations=0;int validateState(DeviceState*s,const char*){++validations;return s?0:1;}
std::string error;void setLastErrorText(const char*p){error=p;}int gasCalls=0;
int ugkwpGpuResidentStrictAdvanceGasOnly(void*,double dt,double time){assert(dt==.1&&time==3);++gasCalls;return 0;}
int combined(void*handle,double dt,double simulationTime){BODY return 73;}
int main(){
    assert(combined(nullptr,.1,3)==1&&gasCalls==0&&validations==1);
    DeviceState s;
    for(auto mode:{ugkwp::GasMode::MixtureFrozen,ugkwp::GasMode::MixtureChemistry})
    for(int capacity:{0,100}){
        s.gasSpecies.mode=mode;s.particleCapacity=capacity;s.particlesMayBePresent=false;
        const int before=gasCalls;
        assert(combined(&s,.1,3)==0&&gasCalls==before+1);
        s.particlesMayBePresent=true;error.clear();
        assert(combined(&s,.1,3)!=0&&gasCalls==before+1);
        assert(error.find("no implementation")!=std::string::npos);
    }
    s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s.particlesMayBePresent=false;
    assert(combined(&s,.1,3)==73&&gasCalls==4&&validations==10);
}
'''.replace('BODY', body)
    source = tmp_path / 'routing.cpp'
    source.write_text(code)
    binary = tmp_path / 'routing'
    result = subprocess.run(['g++', '-std=c++17', '-I' + str(ROOT), str(source), '-o', str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert subprocess.run([str(binary)]).returncode == 0
