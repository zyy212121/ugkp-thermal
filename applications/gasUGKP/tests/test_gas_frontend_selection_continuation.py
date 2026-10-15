"""Selected existing native constructors and capability paths remain reachable."""
from pathlib import Path
import os
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[3]
@pytest.mark.parametrize('relative',[
 'applications/gasUGKP/gpu/GpuDragModel.H',
 'examples/consistency/dustyWave/assets/solver_source/gpu/GpuDragModel.H',
 'examples/consistency/windSandShockTube/assets/solver_source/gpu/GpuDragModel.H'])
def test_t07_real_drag_constructors_preserve_signed_parameters(tmp_path,relative):
    if not os.environ.get('WM_PROJECT_DIR'):pytest.skip('OpenFOAM environment required')
    source=f'''#include "{ROOT/relative}"
#include "IStringStream.H"
#include <iostream>
#include <iterator>
int main(){{std::string text((std::istreambuf_iterator<char>(std::cin)),{{}});Foam::IStringStream is(text);Foam::dictionary dict(is);auto model=Foam::GpuDragModel::New(dict);const auto&p=model->parameters();std::cout<<model->modelId()<<" "<<p[0]<<" "<<p[1]<<" "<<p[2]<<"\\n";}}
'''
    (tmp_path/'Make').mkdir();(tmp_path/'probe.C').write_text(source)
    (tmp_path/'Make/files').write_text(f'probe.C\nEXE = {tmp_path}/probe\n')
    (tmp_path/'Make/options').write_text('EXE_INC =\nEXE_LIBS = -lOpenFOAM\n')
    built=subprocess.run(['wmake'],cwd=tmp_path,capture_output=True,text=True)
    assert built.returncode==0,built.stdout+built.stderr
    def run(text):return subprocess.run([str(tmp_path/'probe')],input=text,capture_output=True,text=True)
    ordinary=run('dragModel GidaspowErgunWenYu;GidaspowErgunWenYuCoeffs {residualRe 0.01;}')
    assert ordinary.returncode==0 and ordinary.stdout.strip()=='1 0.01 0 0',ordinary.stderr
    negative=run('dragModel GidaspowErgunWenYu;GidaspowErgunWenYuCoeffs {residualRe -0.01;}')
    assert negative.returncode==0 and negative.stdout.strip()=='1 -0.01 0 0',negative.stderr
    if relative.startswith('examples'):
        constant=run('dragModel constantResponseTime;gpuResidentParticleResponseTime -2;gpuResidentParticleThermalResponseTime -3;gpuResidentGasHeatCapacity -4;gpuResidentHeatTransferModel unrelated;')
        assert constant.returncode==0 and constant.stdout.strip()=='2 -2 -3 -4',constant.stderr
    for bad in ['dragModel unknown;','dragModel GidaspowErgunWenYu;GidaspowErgunWenYuCoeffs {}']:
        failure=run(bad);assert failure.returncode!=0,failure.stdout


def test_t07_capabilities_allow_executable_combinations_but_no_model_substitution(tmp_path):
    source=tmp_path/'probe.cpp';binary=tmp_path/'probe'
    source.write_text(r'''
#include "common/gasTransport/GasCapabilities.H"
#include <cassert>
int main(){using namespace ugkwp;GasCapabilityRequest r;
 r.movingGeometry=true;r.timeIntegrator=3;assert(validateGasCapabilities(r));
 r.timeIntegrator=1;r.reconstruction=2;assert(validateGasCapabilities(r));
 r.mode=GasMode::MixtureChemistry;r.reconstruction=1;r.turbulenceModel=3;r.sstWallTreatment=1;assert(validateGasCapabilities(r));
 r.movingGeometry=false;r.fluxScheme=4;assert(validateGasCapabilities(r));
 r.fluxScheme=1;r.reconstruction=2;assert(validateGasCapabilities(r));
 r.reconstruction=1;r.particleCoupling=true;assert(!validateGasCapabilities(r));
 r.particleCoupling=false;r.sstWallTreatment=2;r.boundaryLayerModel=0;r.turbulenceModel=3;assert(validateGasCapabilities(r));
 r.boundaryLayerModel=1;r.turbulenceModel=1;assert(validateGasCapabilities(r));
 r.fluxScheme=99;assert(validateGasCapabilities(r));
}
''')
    build=subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
